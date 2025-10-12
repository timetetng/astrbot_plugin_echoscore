import base64
import httpx
from io import BytesIO
from PIL import Image
from typing import List, Optional, Dict, Any

from astrbot.api import logger, AstrBotConfig
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
import astrbot.api.message_components as Comp

async def get_image_from_event(event: AstrMessageEvent) -> List[Comp.Image]:
    """
    从事件中提取图片组件。
    支持直接发送的图片和引用回复中的图片。
    """
    images = []
    
    # 从当前消息中直接提取图片
    if hasattr(event, 'message_obj') and event.message_obj and hasattr(event.message_obj, 'message'):
        for component in event.message_obj.message:
            if isinstance(component, Comp.Image):
                images.append(component)
            # 处理引用/回复中的图片
            elif isinstance(component, Comp.Reply):
                replied_chain = getattr(component, 'chain', None)
                if replied_chain:
                    for reply_comp in replied_chain:
                        if isinstance(reply_comp, Comp.Image):
                            logger.info("从引用消息中获取到图片")
                            images.append(reply_comp)
    
    # 去重
    unique_images = []
    seen = set()
    for img in images:
        identifier = img.url or img.file or id(img.raw)
        if identifier and identifier not in seen:
            unique_images.append(img)
            seen.add(identifier)
            
    return unique_images


@register(
    "astrbot_plugin_echoscore",
    "loping151 & timetetng",
    "基于loping151识别《鸣潮》声骸评分API的astrbot插件，提供LLM交互和指令两种使用方式",
    "2.0.1", # 版本号迭代
    "https://github.com/timetetng/astrbot_plugin_echoscore"
)
class ScoreEchoPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        # 使用 httpx.AsyncClient 作为推荐的异步请求库
        self.http_client = httpx.AsyncClient(timeout=30.0)

    async def _perform_scoring(self, command_str: str, images: List[Comp.Image]) -> Dict[str, Any]:
        """
        执行评分的核心逻辑，包括图片处理和API请求。
        返回一个包含结果或错误的字典。
        """
        images_b64 = []
        try:
            for img_comp in images:
                image_bytes = None
                if img_comp.url:
                    resp = await self.http_client.get(img_comp.url)
                    resp.raise_for_status()
                    image_bytes = resp.content
                elif img_comp.file:
                    with open(img_comp.file, 'rb') as f:
                        image_bytes = f.read()
                elif img_comp.raw:
                    image_bytes = img_comp.raw
                else:
                    continue
                
                # 图片压缩处理
                MAX_SIZE_BYTES = 2 * 1024 * 1024  # 2MB
                with Image.open(BytesIO(image_bytes)) as img:
                    if img.mode not in ("RGB"):
                        img = img.convert("RGB")

                    output_buffer = BytesIO()
                    quality = 95
                    
                    # 循环压缩直到小于目标大小
                    while quality > 10:
                        output_buffer.seek(0)
                        output_buffer.truncate()
                        img.save(output_buffer, format="WEBP", quality=quality)
                        if output_buffer.tell() < MAX_SIZE_BYTES:
                            break
                        quality -= 5
                    
                    compressed_image_bytes = output_buffer.getvalue()

                images_b64.append(base64.b64encode(compressed_image_bytes).decode('utf-8'))
        
        except httpx.RequestError as e:
            logger.error(f"下载图片失败: {e}")
            return {"success": False, "error": f"图片下载失败: {e}"}
        except Exception as e:
            logger.error(f"处理图片时发生错误: {e}")
            return {"success": False, "error": f"图片处理失败: {e}"}
            
        if not images_b64:
            return {"success": False, "error": "未能成功处理任何有效图片。"}

        # 从配置中获取 Token 和 API 地址
        api_token = self.config.get("xwtoken", "your_token_here")
        api_endpoint = self.config.get("endpoint", "https://scoreecho.loping151.site/score")

        if api_token == "your_token_here":
            return {"success": False, "error": "请在插件配置中填写你的 xwtoken！"}

        headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json"
        }
        payload = {
            "command_str": command_str,
            "images_base64": images_b64
        }
        
        try:
            response = await self.http_client.post(api_endpoint, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            
            result_image_b64 = data.get("result_image_base64")
            if result_image_b64:
                return {"success": True, "image_base64": result_image_b64}
            else:
                message = data.get("message", "API未能生成图片，可能是不支持的图片格式或内容。")
                return {"success": False, "error": message}

        except httpx.HTTPStatusError as e:
            error_msg = f"API请求失败，服务器返回错误码: {e.response.status_code}"
            try:
                error_detail = e.response.json().get("detail", "无详细信息")
                error_msg += f"\n错误信息: {error_detail}"
            except Exception:
                error_msg += f"\n原始响应: {e.response.text}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
        except httpx.RequestError as e:
            logger.exception(f"连接评分服务器失败: {e}")
            return {"success": False, "error": f"网络请求失败: {e}"}
        except Exception as e:
            logger.exception(f"执行评分时发生未知错误: {e}")
            return {"success": False, "error": f"未知错误: {e}"}

    @filter.command("评分", alias={'echo', 'score'})
    async def score_command_handler(
        self, event: AstrMessageEvent, role: Optional[str] = None, 
        cost: Optional[str] = None, main_stat: Optional[str] = None
    ):
        """
        处理声骸评分请求的指令。
        指令: /评分 [角色名] [cost] [主词条] (所有参数可选)
        """
        command_str = event.message_str.strip().split("评分", 1)[-1].strip()
        
        images = await get_image_from_event(event)
        if not images:
            yield event.plain_result("请在发送命令的同时附带需要评分的声骸截图，或者回复一张声骸截图哦~")
            return

        yield event.plain_result(f"收到 {len(images)} 张图片，正在分析中，请稍候...")
        
        result = await self._perform_scoring(command_str, images)

        if result["success"]:
            base64_uri = f"base64://{result['image_base64']}"
            yield event.chain_result([Comp.Image(file=base64_uri)])
        else:
            yield event.plain_result(f"评分失败了：\n{result['error']}")

    @filter.llm_tool(name="score_wuthering_waves_echo")
    async def score_wuthering_waves_echo(
        self, event: AstrMessageEvent, role: Optional[str] = None, 
        cost: Optional[str] = None, main_stat: Optional[str] = None
    ) -> str:
        """
        对《鸣潮》游戏中的声骸截图进行评分。你必须从当前对话或用户的引用中获取图片来进行评分。
        Args:
            role (string): 角色中文名，例如'忌炎'或'吟霖'。如果用户没有指定，则忽略。
            cost (string): 声骸的COST值，例如'4c', '3c', '1c'。如果用户没有指定，则忽略。
            main_stat (string): 声骸的主词条，例如'暴击'或'攻击'。如果用户没有指定，则忽略。
        """
        logger.info(f"LLM tool 'score_wuthering_waves_echo' called with: role={role}, cost={cost}, main_stat={main_stat}")
        
        parts = [p for p in [role, cost, main_stat] if p]
        command_str = " ".join(parts)
        
        images = await get_image_from_event(event)
        if not images:
            return "评分失败，因为我没有在当前对话中找到任何图片。请提醒用户需要发送一张声骸截图。"
        
        result = await self._perform_scoring(command_str, images)

        if result["success"]:
            base64_uri = f"base64://{result['image_base64']}"
            await event.send(event.chain_result([Comp.Image(file=base64_uri)]))
            return "评分图片已成功发送给用户。"
        else:
            return f"评分失败了，请将以下原因告知用户：{result['error']}"

    async def terminate(self):
        await self.http_client.aclose()
        logger.info("ScoreEchoPlugin http client closed.")