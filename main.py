import asyncio
import base64
import json
import re
import time
from io import BytesIO
from typing import Any

import httpx
from PIL import Image

import astrbot.api.message_components as Comp
from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star, register


async def get_image_from_direct_event(event: AstrMessageEvent) -> list[Comp.Image]:
    """从当前事件中提取所有图片组件，这是最高优先级的图片来源。"""
    images = []
    try:
        # 尝试直接访问 event.message_obj.message
        for component in event.message_obj.message:
            if isinstance(component, Comp.Image):
                images.append(component)
            elif isinstance(component, Comp.Reply):
                replied_chain = getattr(component, "chain", None)
                if replied_chain:
                    for reply_comp in replied_chain:
                        if isinstance(reply_comp, Comp.Image):
                            images.append(reply_comp)
    except AttributeError:
        pass

    unique_images = []
    seen = set()
    for img in images:
        identifier = img.url or img.file
        if identifier and identifier not in seen:
            unique_images.append(img)
            seen.add(identifier)
        elif not identifier:
            unique_images.append(img)
    return unique_images


@register(
    "astrbot_plugin_echoscore",
    "loping151 & timetetng",
    "基于loping151识别《鸣潮》声骸评分API的astrbot插件，提供LLM交互和指令两种使用方式",
    "3.1.3",
    "https://github.com/timetetng/astrbot_plugin_echoscore",
)
class ScoreEchoPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.http_client = httpx.AsyncClient(timeout=30.0)

        # --- 从配置初始化缓存和别名设置 ---
        self.enable_context_cache = self.config.get("enable_context_cache", True)
        self.context_cache_size = self.config.get("context_cache_size", 3)
        self.enable_alias_mapping = self.config.get("enable_alias_mapping", False)
        self.alias_file_path = self.config.get("alias_file_path", "")
        self.alias_map: dict[str, str] = {}

        self.submission_window_seconds = 60
        self.context_image_cache: dict[str, tuple[list[Comp.Image], float]] = {}

        logger.info("鸣潮声骸评分插件加载成功")
        logger.info(
            f"上下文图片缓存: {'已开启' if self.enable_context_cache else '已关闭'}, 缓存数量: {self.context_cache_size}"
        )
        logger.info(
            f"角色别名映射: {'已开启' if self.enable_alias_mapping else '已关闭'}"
        )

        # --- 加载别名文件并构建反向映射表 ---
        if self.enable_alias_mapping and self.alias_file_path:
            try:
                with open(self.alias_file_path, encoding="utf-8") as f:
                    alias_data = json.load(f)

                # 构建 {别名: 本名} 的反向映射，方便快速查找
                for canonical_name, aliases in alias_data.items():
                    for alias in aliases:
                        self.alias_map[alias.lower()] = (
                            canonical_name  # 使用小写以忽略大小写
                        )

                logger.info(
                    f"成功加载角色别名文件，共构建 {len(self.alias_map)} 条别名映射。"
                )

            except FileNotFoundError:
                logger.error(f"角色别名文件未找到，路径: {self.alias_file_path}")
            except json.JSONDecodeError:
                logger.error(
                    f"角色别名文件格式错误，请检查是否为有效的JSON文件: {self.alias_file_path}"
                )
            except Exception as e:
                logger.error(f"加载角色别名文件时发生未知错误: {e}")
        elif self.enable_alias_mapping:
            logger.warning("已启用角色别名映射，但未提供别名文件路径。")

    @filter.event_message_type(filter.EventMessageType.ALL, priority=-1)
    async def on_any_message_for_image_cache(self, event: AstrMessageEvent):
        """监听并缓存所有图片，仅为无直接图片的上下文调用提供支持。"""
        if not self.enable_context_cache:
            return

        umo = event.unified_msg_origin
        images_in_message = [
            comp for comp in event.message_obj.message if isinstance(comp, Comp.Image)
        ]

        if images_in_message:
            now = time.time()
            cached_images, last_update_time = self.context_image_cache.get(umo, ([], 0))
            if now - last_update_time > self.submission_window_seconds:
                cached_images = []
            cached_images.extend(images_in_message)
            if self.context_cache_size > 0:
                cached_images = cached_images[-self.context_cache_size :]
            else:
                cached_images = []

            self.context_image_cache[umo] = (cached_images, now)
            if cached_images:
                logger.info(
                    f"[EchoScore Cache] 已更新会话 {umo} 的上下文缓存，当前共 {len(cached_images)} 张图片。"
                )

    async def _get_images_from_context(
        self, event: AstrMessageEvent
    ) -> list[Comp.Image]:
        images = await get_image_from_direct_event(event)
        if images:
            logger.info(f"图片获取：从直接消息/引用中找到 {len(images)} 张图片。")
            return images

        if not images and self.enable_context_cache:
            umo = event.unified_msg_origin
            cached_data = self.context_image_cache.get(umo)
            if cached_data:
                images = cached_data[0]
                logger.info(
                    f"图片获取：从上下文缓存为会话 {umo} 找到了 {len(images)} 张图片。"
                )

        return images

    @staticmethod
    def _process_image(image_bytes: bytes) -> bytes:
        """
        处理图片：转换RGB、调整大小和压缩。
        将被 asyncio.to_thread 放到线程池中执行，避免阻塞主线程。
        """
        MAX_SIZE_BYTES = 2 * 1024 * 1024
        with Image.open(BytesIO(image_bytes)) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            output_buffer = BytesIO()
            quality = 95
            while quality > 10:
                output_buffer.seek(0)
                output_buffer.truncate()
                img.save(output_buffer, format="WEBP", quality=quality)
                if output_buffer.tell() < MAX_SIZE_BYTES:
                    break
                quality -= 5
            return output_buffer.getvalue()

    async def _perform_scoring(
        self, command_str: str, images: list[Comp.Image]
    ) -> dict[str, Any]:
        images_b64 = []
        try:
            # 1. 准备图片处理任务
            tasks = []
            for img_comp in images:
                image_b64_str = await img_comp.convert_to_base64()
                if not image_b64_str:
                    continue
                image_bytes = base64.b64decode(image_b64_str)
                tasks.append(asyncio.to_thread(self._process_image, image_bytes))

            if not tasks:
                return {"success": False, "error": "未能成功获取任何有效图片。"}

            # 2. 并发执行所有图片处理任务
            processed_images_bytes = await asyncio.gather(*tasks)

            # 3. 将处理后的结果编码回 base64
            for compressed_bytes in processed_images_bytes:
                images_b64.append(base64.b64encode(compressed_bytes).decode("utf-8"))

        except Exception as e:
            return {"success": False, "error": f"图片处理失败: {e}"}

        if not images_b64:
            return {"success": False, "error": "图片处理后为空，无法评分。"}

        api_token = self.config.get("xwtoken", "your_token_here")

        # 强制检查 xwtoken，如果无效则直接报错
        if not api_token or api_token == "your_token_here":
            return {
                "success": False,
                "error": "未配置有效的 xwtoken。请在插件配置中填写有效的 Token 以使用评分服务。",
            }

        # 直接使用认证端点
        api_endpoint = self.config.get(
            "endpoint", "https://scoreecho.loping151.site/score"
        )
        headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
        }

        payload = {"command_str": command_str, "images_base64": images_b64}

        try:
            response = await self.http_client.post(
                api_endpoint, headers=headers, json=payload
            )
            response.raise_for_status()
            data = response.json()
            if data.get("result_image_base64"):
                return {"success": True, "image_base64": data["result_image_base64"]}
            else:
                return {
                    "success": False,
                    "error": data.get("message", "API未能生成图片。"),
                }
        except httpx.HTTPStatusError as e:
            return {
                "success": False,
                "error": f"API请求返回错误状态码: {e.response.status_code} - {e}",
            }
        except Exception as e:
            return {"success": False, "error": f"API请求或处理时发生错误: {e}"}

    @filter.command("清空评分缓存")
    async def clear_score_cache_command(self, event: AstrMessageEvent):
        umo = event.unified_msg_origin
        if umo in self.context_image_cache:
            del self.context_image_cache[umo]
            yield event.plain_result("当前会話的声骸评分图片缓存已清空！")
        else:
            yield event.plain_result("当前会話没有需要清空的缓存哦~")

    # --- 辅助函数，用于解析角色别名 ---
    def _resolve_role_alias(self, role_input: str | None) -> str | None:
        """根据加载的别名表解析角色名"""
        if not self.enable_alias_mapping or not role_input:
            return role_input  # 如果功能关闭或输入为空，直接返回原值

        # 使用.get()方法，如果找不到别名，则返回原输入值
        resolved_name = self.alias_map.get(role_input.lower(), role_input)
        if resolved_name != role_input:
            logger.info(f"角色别名解析: '{role_input}' -> '{resolved_name}'")
        return resolved_name

    # --- LLM 钩子 (实现无空格指令解析) ---
    @filter.on_llm_request(priority=1919810)  # 设置一个高优先级，确保拦截 hook
    async def on_llm_request_handler(
        self, event: AstrMessageEvent, req: ProviderRequest
    ):
        """
        监听 LLM 请求，拦截评分命令（如 /评分千咲, @bot 声骸4c 等）。
        如果匹配到命令，则停止 LLM 请求并自行处理评分。
        """
        text = event.message_str.strip()

        # 正则匹配
        pattern = r"^(评分|查分|声骸|生蚝)\s*(.*)$"
        match = re.match(pattern, text, re.IGNORECASE)

        if not match:
            return  # 未匹配到评分命令，放行给 LLM 处理

        # 匹配成功，拦截事件，阻止 LLM 生成
        event.stop_event()

        args_str = match.group(2)
        role = None
        cost = None
        main_stat = None

        # 1. 提取 Cost (支持 1c, 3c, 4c, c1, c3, c4, 忽略大小写)
        cost_match = re.search(r"([134]c|c[134])", args_str, re.IGNORECASE)
        if cost_match:
            cost = cost_match.group(1)
            # 从参数字符串中移除 cost，剩余部分作为角色名/主词条
            args_str = args_str.replace(cost, "").strip()

        # 2. 提取角色名和主词条
        # 如果剩余字符串包含空格，尝试分割为 角色 和 主词条
        # 如果没有空格，则认为剩余部分全是角色名
        if args_str:
            parts = args_str.split(maxsplit=1)
            role = parts[0]
            if len(parts) > 1:
                main_stat = parts[1]

        # 3. 获取图片
        images = await self._get_images_from_context(event)
        if not images:
            await event.send(
                event.plain_result(
                    "请在发送命令的同时附带声骸截图，或回复包含截图的消息。"
                )
            )
            return

        # 4. 别名解析
        resolved_role = self._resolve_role_alias(role)

        # 5. 组合 API 请求字符串
        cmd_parts = [p for p in [resolved_role, cost, main_stat] if p]
        command_str = " ".join(cmd_parts)

        # 6. 清理缓存
        umo = event.unified_msg_origin
        if umo in self.context_image_cache:
            del self.context_image_cache[umo]
            logger.info(f"指令任务开始，已清除会话 {umo} 的上下文缓存。")

        # 7. 执行评分
        result = await self._perform_scoring(command_str, images)

        if result["success"]:
            await event.send(
                event.chain_result(
                    [Comp.Image(file=f"base64://{result['image_base64']}")]
                )
            )
        else:
            await event.send(event.plain_result(f"评分失败了：\n{result['error']}"))

    # --- LLM 工具层 ---
    @filter.llm_tool(name="score_wuthering_waves_echo")
    async def score_wuthering_waves_echo(
        self,
        event: AstrMessageEvent,
        role: str | None = None,
        cost: str | None = None,
        main_stat: str | None = None,
    ) -> str:
        """
        请在用户询问有关鸣潮角色声骸评分时使用。
        本工具可以处理单张或多张图片，支持对比评分任务。
        用户往往会说xx1c、xx3c等来指定参数，其中xx是角色名，1c/3c/4c是cost。
        Args:
            role (string): 角色中文名。如果用户没有指定，则忽略。
            cost (string): 声骸的COST值，例如'4c', '3c', '1c'。如果用户没有指定，则忽略。
            main_stat (string): 声骸的主词条，例如'暴击'或'攻击'。如果用户没有指定，则忽略。
        """
        logger.info(
            f"LLM tool 'score_wuthering_waves_echo' called with: role={role}, cost={cost}, main_stat={main_stat}"
        )

        images = await self._get_images_from_context(event)

        if not images:
            return "评分失败，我没有在用户的消息或上下文中找到任何图片。请提醒用户需要发送一张声骸截图。"

        umo = event.unified_msg_origin
        if umo in self.context_image_cache:
            del self.context_image_cache[umo]
            logger.info(f"LLM任务开始，已清除会话 {umo} 的上下文缓存。")

        # --- 解析角色别名 ---
        resolved_role = self._resolve_role_alias(role)

        parts = [p for p in [resolved_role, cost, main_stat] if p]
        command_str = " ".join(parts)
        result = await self._perform_scoring(command_str, images)

        if result["success"]:
            await event.send(
                event.chain_result(
                    [Comp.Image(file=f"base64://{result['image_base64']}")]
                )
            )
            return f"{resolved_role}的声骸评分图片已发送，请继续和用户正常回复。"
        else:
            return f"{resolved_role}的声骸评分失败了，请将以下原因告知用户：{result['error']}"

    async def terminate(self):
        await self.http_client.aclose()
        logger.info("鸣潮声骸评分插件已关闭")  #
