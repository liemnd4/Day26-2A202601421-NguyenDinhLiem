"""Minh hoạ FUNCTION CALLING thuần với Google Gemini SDK.

Tool `get_weather` được định nghĩa schema thủ công VÀ thực thi ngay trong
chính file app này. Model chỉ QUYẾT ĐỊNH gọi tool nào; app mới là nơi chạy.

Cách chạy:
    pip install -r ../requirements.txt
    export GEMINI_API_KEY=...
    python weather_function_calling.py
"""

import json
import os

from google import genai
from google.genai import types

API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "Thiếu API key. Đặt biến môi trường GEMINI_API_KEY hoặc GOOGLE_API_KEY."
    )

client = genai.Client(api_key=API_KEY)

MODEL = "gemini-2.5-flash"

SYSTEM_INSTRUCTION = (
    "Bạn là trợ lý thời tiết thân thiện, trả lời bằng tiếng Việt tự nhiên. "
    "Dùng emoji phù hợp (🌧️ 🌤️ 💨 💧). "
    "Tóm tắt ngắn gọn, dễ hiểu, và đưa ra lời khuyên thực tế "
    "(ví dụ: mang ô, mặc áo mỏng, ...)."
)

# 1. App tự định nghĩa schema của tool
get_weather_declaration = types.FunctionDeclaration(
    name="get_weather",
    description="Lấy thời tiết hiện tại của một thành phố",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "city": types.Schema(
                type=types.Type.STRING, description="Tên thành phố"
            )
        },
        required=["city"],
    ),
)

get_forecast_declaration = types.FunctionDeclaration(
    name="get_forecast",
    description="Lấy dự báo thời tiết nhiều ngày của một thành phố",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "city": types.Schema(
                type=types.Type.STRING, description="Tên thành phố"
            ),
            "days": types.Schema(
                type=types.Type.INTEGER,
                description="Số ngày dự báo, từ 1 đến 3. Mặc định 3.",
            ),
        },
        required=["city"],
    ),
)

TOOLS = [
    types.Tool(
        function_declarations=[get_weather_declaration, get_forecast_declaration]
    )
]


# 2. App tự thực thi tool (trong thực tế sẽ gọi API thời tiết thật)
def get_weather(city: str) -> str:
    """Trả về thời tiết (mock) của *city*. Dùng làm tool cho model."""
    mock_data = {
        "Hà Nội": {
            "nhiệt_độ": "29°C",
            "thời_tiết": "trời mưa nhẹ",
            "độ_ẩm": "82%",
            "gió": {"hướng": "Đông Nam", "tốc_độ": "12 km/h"},
        },
        "Hồ Chí Minh": {
            "nhiệt_độ": "33°C",
            "thời_tiết": "mưa rào",
            "độ_ẩm": "75%",
            "gió": {"hướng": "Tây Nam", "tốc_độ": "15 km/h"},
        },
        "Đà Nẵng": {
            "nhiệt_độ": "30°C",
            "thời_tiết": "nhiều mây",
            "độ_ẩm": "78%",
            "gió": {"hướng": "Đông", "tốc_độ": "10 km/h"},
        },
    }
    default = {"nhiệt_độ": "28°C", "thời_tiết": "không có dữ liệu chi tiết"}
    return json.dumps({"city": city, **mock_data.get(city, default)}, ensure_ascii=False)


def get_forecast(city: str, days: int = 3) -> str:
    """Trả về dự báo mock 1–3 ngày cho *city* để model dùng làm tool."""
    days = max(1, min(days, 3))
    forecast_data = {
        "Hà Nội": [
            {"ngày": "Hôm nay", "nhiệt_độ": "29°C", "thời_tiết": "mưa nhẹ"},
            {"ngày": "Ngày mai", "nhiệt_độ": "30°C", "thời_tiết": "nhiều mây"},
            {"ngày": "Ngày kia", "nhiệt_độ": "31°C", "thời_tiết": "nắng gián đoạn"},
        ],
        "Hồ Chí Minh": [
            {"ngày": "Hôm nay", "nhiệt_độ": "33°C", "thời_tiết": "mưa rào"},
            {"ngày": "Ngày mai", "nhiệt_độ": "32°C", "thời_tiết": "mưa chiều"},
            {"ngày": "Ngày kia", "nhiệt_độ": "33°C", "thời_tiết": "có mây"},
        ],
        "Đà Nẵng": [
            {"ngày": "Hôm nay", "nhiệt_độ": "30°C", "thời_tiết": "nhiều mây"},
            {"ngày": "Ngày mai", "nhiệt_độ": "31°C", "thời_tiết": "nắng nhẹ"},
            {"ngày": "Ngày kia", "nhiệt_độ": "30°C", "thời_tiết": "mưa rào"},
        ],
    }
    default = [
        {"ngày": "Hôm nay", "nhiệt_độ": "28°C", "thời_tiết": "chưa có dữ liệu chi tiết"},
        {"ngày": "Ngày mai", "nhiệt_độ": "29°C", "thời_tiết": "chưa có dữ liệu chi tiết"},
        {"ngày": "Ngày kia", "nhiệt_độ": "29°C", "thời_tiết": "chưa có dữ liệu chi tiết"},
    ]
    return json.dumps(
        {"city": city, "forecast": forecast_data.get(city, default)[:days]},
        ensure_ascii=False,
    )


TOOL_HANDLERS = {"get_weather": get_weather, "get_forecast": get_forecast}


def run(prompt: str) -> str:
    """Gửi *prompt* tới Gemini, tự động xử lý function calling và trả về câu trả lời cuối."""
    contents: list[types.Content] = [
        types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
    ]

    # 3. Gọi model — model quyết định có gọi tool hay không
    resp = client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            tools=TOOLS,
            system_instruction=SYSTEM_INSTRUCTION,
        ),
    )

    # 4. Vòng lặp: nếu model yêu cầu tool, app TỰ THỰC THI rồi đưa kết quả trả lại
    while resp.function_calls:
        # Thêm phản hồi của model vào lịch sử hội thoại
        contents.append(resp.candidates[0].content)

        function_responses = []
        for fc in resp.function_calls:
            print(f"  [model yêu cầu] {fc.name}({fc.args})")
            handler = TOOL_HANDLERS.get(fc.name)
            if handler is None:
                result = json.dumps({"error": f"Tool không hỗ trợ: {fc.name}"})
            else:
                result = handler(**fc.args)  # <-- app chạy, không phải model
            print(f"  [app thực thi]  -> {result}")
            function_responses.append(
                types.Part.from_function_response(
                    name=fc.name, response={"result": result}
                )
            )

        # Gửi kết quả tool trả về cho model
        contents.append(types.Content(role="user", parts=function_responses))
        resp = client.models.generate_content(
            model=MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                tools=TOOLS,
                system_instruction=SYSTEM_INSTRUCTION,
            ),
        )

    # 5. Model tổng hợp câu trả lời cuối
    return resp.text


if __name__ == "__main__":
    question = "Dự báo thời tiết Hà Nội trong 3 ngày tới thế nào?"
    print(f"User: {question}\n")
    print("Trả lời:", run(question))
