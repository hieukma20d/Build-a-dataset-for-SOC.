import httpx
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, root_validator
from typing import Optional, List, Any
import uvicorn
import requests
import json
import re
import urllib3
import time

# ==========================================
# CHỈNH CẤU HÌNH HỆ THỐNG TẠI ĐÂY
# ==========================================
OLLAMA_API_URL = "http://localhost:11434/api/generate"
# Sử dụng model đã được tinh chỉnh hoặc prompt engineer tốt cho tác vụ SOC
MODEL_NAME = "llama31-soc:latest" 

# Cấu hình Splunk HEC (HTTP Event Collector)
SPLUNK_HEC_URL = "https://localhost:8088/services/collector/event"
SPLUNK_HEC_TOKEN = "e2075e20-648f-4adc-8278-d5a23b41e321" # TOKEN CỦA BẠN
TARGET_INDEX = "cic_ids" # Index đồ án

# ==========================================

# Ẩn cảnh báo SSL tự ký của Splunk
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

app = FastAPI(
    title="KMA CyberSecurity SOC LLM Internal API - NL2Query Edition",
    description="API Gateway nội bộ tích hợp module NL2Query (AI tự sinh câu lệnh SPL/KQL)",
    version="5.0.0"
)

# --- 1. PYDANTIC MODELS (Chuẩn hóa dữ liệu) ---

class SplunkResultData(BaseModel):
    """Bóc tách dữ liệu mạng thô từ cảnh báo Splunk"""
    raw_log: str = Field(..., alias="_raw")
    protocol: str = Field(..., description="Giao thức tầng truyền tải (TCP/UDP)")
    dest_port: int = Field(..., description="Cổng đích dịch vụ")
    flow_duration: Any = Field(..., description="Thời gian kéo dài luồng")
    packet_count: int = Field(..., description="Tổng số gói tin")
    tcp_flags: Optional[str] = Field(None, description="Cờ điều khiển TCP")
    host: Optional[str] = "unknown_host"
    src_ip: Optional[str] = Field("unknown_src", alias="src_ip")

    class Config:
        populate_by_name = True

class SplunkWebhookPayload(BaseModel):
    """Payload nhận từ Splunk Webhook Alert"""
    search_name: str = Field(..., description="Tên bẫy cảnh báo")
    sid: str = Field(..., description="Mã định danh sự cố")
    result: SplunkResultData = Field(..., alias="result")

class SOCAnalysisResponse(BaseModel):
    """Cấu trúc phản hồi JSON sạch do AI tự sinh"""
    severity: str = Field(..., description="Mức độ nghiêm trọng (High/Medium/Low)")
    attack_type: str = Field(..., description="Phân loại tấn công tiếng Việt")
    findings: List[str] = Field(..., description="Nhận định kỹ thuật tiếng Việt")
    recommended_actions: List[str] = Field(..., description="Cẩm nang ứng cứu tiếng Việt")
    generated_query: str = Field(..., description="Câu lệnh SPL do AI tự sinh")
    generated_kql: Optional[str] = Field("DeviceNetworkEvents | take 10", description="Câu lệnh KQL do AI tự sinh")

# --- 2. HÀM BỔ TRỢ (Utility Functions) ---

def clean_hallucinated_text(text: str) -> str:
    """Loại bỏ dấu ba chấm lửng lơ và token rác thường gặp ở local model"""
    if not text: return ""
    # Xóa dấu ... ở cuối câu
    text = re.sub(r'\s*\.\.\.+\s*$', '.', text.strip())
    # Sửa một số lỗi font/từ vựng phổ biến
    text = text.replace("pecypc", "tài nguyên")
    return text

def push_to_splunk_hec(analysis_result: dict, original_sid: str):
    """Đẩy bản ghi JSON đã qua AI phân tích và sinh lệnh về Splunk HEC"""
    headers = {'Authorization': f'Splunk {SPLUNK_HEC_TOKEN}'}
    payload = {
        "event": {
            "original_sid": original_sid,
            "analysis": analysis_result,
            "status": "ai_analyzed_nl2query",
            "analyst_team": "KMA_SOC_AI_BOT"
        },
        "index": TARGET_INDEX,
        "sourcetype": "_json"
    }
    try:
        r = requests.post(SPLUNK_HEC_URL, json=payload, headers=headers, verify=False, timeout=10)
        if r.status_code == 200:
            print(f"[*] SID {original_sid}: Đã đẩy kết quả NL2Query về Splunk HEC.")
        else:
            print(f"[!] HEC từ chối: {r.status_code} - {r.text}")
    except Exception as e:
        print(f"[!] Lỗi kết nối Splunk HEC: {e}")

async def call_local_ollama(system_prompt: str, user_prompt: str) -> str:
    """Gọi Ollama API cục bộ (bất đồng bộ) với cấu hình ép khuôn JSON"""
    ollama_payload = {
        "model": MODEL_NAME,
        "prompt": f"{system_prompt}\n\nUser Input:\n{user_prompt}",
        "stream": False,
        "format": "json", # Ép AI nhả JSON thuần
        "options": {
            "temperature": 0.3, # Thấp để đảm bảo tính chính xác của câu lệnh
            "repeat_penalty": 1.2,
            "num_predict": 1024,
            "stop": ["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "```"]
        }
    }
    
    start_time = time.time()
    async with httpx.AsyncClient(timeout=180.0) as client:
        try:
            response = await client.post(OLLAMA_API_URL, json=ollama_payload)
            response.raise_for_status()
            result_json = response.json()
            model_output = result_json.get("response", "{}").strip()
            
            # Làm sạch markdown nếu AI lỡ tay viết vào
            if model_output.startswith("```"):
                model_output = re.sub(r'^```json\s*|```\s*$', '', model_output, flags=re.MULTILINE)
            
            end_time = time.time()
            print(f"[*] Thời gian AI suy luận và sinh lệnh: {end_time - start_time:.2f} giây")
            return model_output
            
        except httpx.ReadTimeout:
            print("[!] Ollama bị timeout (suy luận quá lâu).")
            raise
        except Exception as e:
            print(f"[!] Lỗi gọi Ollama: {e}")
            raise

# --- 3. SYSTEM PROMPT DESIGN (Trái tim của Module NL2Query) ---

# Đây là bộ chỉ thị ép khuôn AI đóng vai chuyên gia SOC kiêm công cụ NL2Query
SYSTEM_PROMPT_NL2QUERY = (
    "Bạn là một chuyên gia SOC Blue-team cấp cao kiêm công cụ NL2Query an ninh mạng nâng cao. "
    "Nhiệm vụ của bạn là đọc dữ liệu log thô và ngữ cảnh cảnh báo để tự động thực hiện hai tác vụ song song:\n"
    "TÁC VỤ 1: Phân tích sự cố, đánh giá mức độ, đưa ra nhận định (findings) và cẩm nang ứng cứu (recommended_actions) bằng Tiếng Việt sạch nhiễu.\n"
    "TÁC VỤ 2 (Module NL2Query): TỰ ĐỘNG SINH câu lệnh truy vấn SPL (Splunk) và KQL (Microsoft Defender) chính xác tuyệt đối để nhà phân tích copy mã lệnh điều tra mở rộng ngay lập tức.\n\n"
    
    "[QUY TẮC SINH CÂU LỆNH SPL/KQL CHUẨN Blue-team]:\n"
    "- Dữ liệu của bạn nằm trong index='cic_ids' và sourcetype='_json'. Hãy luôn sử dụng 'index=cic_ids'.\n"
    "- DDoS/SYN Flood: Sinh lệnh thống kê count by src_ip nhắm vào dest_port bị tấn công.\n"
    "- Port Scan: Sinh lệnh thống kê count by dest_port, host để xem phạm vi dò quét.\n"
    "- Heartbleed/Slowloris: Sinh lệnh lọc flow_duration > 5000 để tìm kết nối kéo dài bất thường.\n"
    "- Cú pháp câu lệnh phải chính xác tuyệt đối, không được thiếu dấu gạch đứng (|), dấu ngoặc, hoặc sai tên trường.\n\n"
    
    "[HỌC TỪ CÁC VÍ DỤ MẪU (Few-Shot Prompting)]:\n"
    "Input context: Tên Alert: 'Alert_DDoS', Port đích: 80, Cờ TCP: SYN.\n"
    "Output generated_query: index=cic_ids sourcetype=\"_json\" dest_port=80 | stats count by src_ip\n"
    "Output generated_kql: DeviceNetworkEvents | where BorderPort == 80 | summarize ConnectionCount = count() by RemoteIP\n\n"
    
    "BẮT BUỘC TRẢ VỀ ĐỊNH DẠNG JSON THUẦN THEO MẪU, KHÔNG VIẾT CHỮ THỪA NGOÀI KHỐI JSON:\n"
    "{\n"
    "  \"severity\": \"HIGH/MEDIUM/LOW/CRITICAL\",\n"
    "  \"attack_type\": \"Tên phân loại tấn công tiếng Việt\",\n"
    "  \"findings\": [\"Nhận định 1\", \"Nhận định 2\"],\n"
    "  \"recommended_actions\": [\"Hành động 1\", \"Hành động 2\"],\n"
    "  \"generated_query\": \"CÂU LỆNH SPL BẠN TỰ SINH\",\n"
    "  \"generated_kql\": \"CÂU LỆNH KQL BẠN TỰ SINH\"\n"
    "}"
)

# --- 4. MAIN API ENDPOINT (Luồng xử lý chính) ---

@app.post("/api/v1/soc/splunk-webhook", response_model=SOCAnalysisResponse)
async def receive_splunk_webhook(payload: SplunkWebhookPayload):
    print(f"\n[+] Tiếp nhận Webhook: Tác vụ NL2Query cho '{payload.search_name}' (SID: {payload.sid})")
    
    # 1. Xây dựng User Prompt dựa trên dữ liệu thô (định hướng cho AI tự sinh lệnh)
    user_prompt = f"""
    [NGỮ CẢNH CẢNH BÁO SIEM]:
    - Threat Signature Name: {payload.search_name}
    
    [DỮ LIỆU LƯU LƯỢNG MẠNG THÔ]:
    - Giao thức: {payload.result.protocol}
    - Cổng đích dịch vụ: {payload.result.dest_port}
    - Thời gian luồng: {payload.result.flow_duration} ms
    - Tổng số gói tin: {payload.result.packet_count}
    - Cờ TCP Flags: {payload.result.tcp_flags if payload.result.tcp_flags else "N/A"}
    
    [BẢN GHI LÝ LỊCH GÓI TIN (_raw)]:
    {payload.result.raw_log}
    
    [YÊU CẦU CHO CHUYÊN GIA]:
    Hãy phân tích ngữ cảnh trên, tự suy luận ra bản chất cuộc tấn công, và thực hiện TÁC VỤ NL2QUERY để sinh ra câu lệnh SPL/KQL chuẩn xác giúp phân tích viên SOC điều tra diện rộng lập tức.
    """

    try:
        # 2. Gọi Local LLM suy luận và tự sinh câu lệnh hoàn toàn
        ai_raw_output = await call_local_ollama(system_prompt=SYSTEM_PROMPT_NL2QUERY, user_prompt=user_prompt)
        
        # 3. Trích xuất và giải mã JSON do AI tự sinh
        try:
            structured_data = json.loads(ai_raw_output)
        except json.JSONDecodeError:
            # Cơ chế phòng vệ nếu AI sinh chuỗi lệnh lỗi cú pháp JSON
            # Thử làm sạch mạnh hơn bằng regex để tìm khối { }
            match = re.search(r'\{.*\}', ai_raw_output, re.DOTALL)
            if match:
                structured_data = json.loads(match.group(0))
            else:
                raise # Ném lỗi lên khối ngoại lệ chính

        # 4. Làm sạch văn bản tiếng Việt do AI sinh (khử nhiễu)
        structured_data["findings"] = [clean_hallucinated_text(f) for f in structured_data.get("findings", []) if f]
        structured_data["recommended_actions"] = [clean_hallucinated_text(a) for a in structured_data.get("recommended_actions", []) if a]
        
        # 5. Kiểm tra tính toàn vẹn của câu lệnh SPL do AI sinh (Phòng ảo giác)
        generated_spl = structured_data.get("generated_query", "")
        if "index=cic_ids" not in generated_spl or len(generated_spl) < 20:
            # Nếu AI sinh lệnh ảo giác, FastAPI sẽ tự động override bằng lệnh chuẩn
            print("[!] AI sinh câu lệnh SPL ảo giác/lỗi cú pháp. Kích hoạt luật override cố định.")
            if "DDoS" in payload.search_name:
                structured_data["generated_query"] = f"index=cic_ids sourcetype=\"_json\" dest_port={payload.result.dest_port} | stats count by src_ip"
            else:
                structured_data["generated_query"] = f"index=cic_ids sourcetype=\"_json\" dest_port={payload.result.dest_port}"

        # 6. Xác thực dữ liệu lần cuối qua Pydantic Model
        analysis_result = SOCAnalysisResponse.model_validate(structured_data)
        
        # 7. Đẩy bản ghi JSON sạch kèm lệnh do AI tự sinh về Splunk HEC
        push_to_splunk_hec(analysis_result.model_dump(), payload.sid)
        
        return analysis_result

    except Exception as e:
        # =================================================================
        # 🛡️ CƠ CHẾ FAIL-SAFE (FALLBACK) TUYỆT ĐỐI
        # Đảm bảo dữ liệu demo luôn chuẩn kể cả khi LLM sập hoặc ảo giác
        # =================================================================
        print(f"[!] Kích hoạt chế độ Fail-Safe (Module NL2Query lỗi): {e}")
        
        # Luật bốc query cố định chuẩn 100% (Override ảo giác của AI)
        if "DDoS" in payload.search_name or "PortScan" in payload.search_name:
            fallback_query = f"index=cic_ids sourcetype=\"_json\" dest_port={payload.result.dest_port} | stats count by src_ip"
            fallback_kql = f"DeviceNetworkEvents | where BorderPort == {payload.result.dest_port} | summarize ConnectionCount = count() by RemoteIP"
        elif "Heartbleed" in payload.search_name:
            fallback_query = f"index=cic_ids sourcetype=\"_json\" flow_duration > 5000 | table _time, src_ip, dest_port, flow_duration"
            fallback_kql = f"DeviceNetworkEvents | where BorderPort == {payload.result.dest_port} | where FlowDuration > 5000"
        else:
            fallback_query = f"index=cic_ids sourcetype=\"_json\" dest_port={payload.result.dest_port}"
            fallback_kql = f"DeviceNetworkEvents | where BorderPort == {payload.result.dest_port}"

        fallback_payload = {
            "severity": "HIGH" if payload.result.packet_count > 2000 else "MEDIUM",
            "attack_type": f"Nghi vấn hành vi độc hại hệ thống ({payload.search_name})",
            "findings": [f"Hệ thống Gateway kích hoạt chế độ Fail-Safe do module NL2Query của AI phản hồi lỗi cấu trúc.",
                        f"Dữ liệu mạng thô ghi nhận luồng lưu lượng bất thường khớp với luật '{payload.search_name}'."],
            "recommended_actions": ["Nhà phân tích SOC sử dụng câu lệnh SPL/KQL mẫu an toàn do Gateway cung cấp để điều tra lập tức.",
                                    "Cách ly luồng mạng nghi vấn nếu cờ cờ kết nối vượt ngưỡng an toàn."],
            "generated_query": fallback_query,
            "generated_kql": fallback_kql
        }
        
        fallback_result = SOCAnalysisResponse.model_validate(fallback_payload)
        push_to_splunk_hec(fallback_result.model_dump(), payload.sid)
        return fallback_result

if __name__ == "__main__":
    # Khởi chạy API Gateway nội bộ trên cổng 8080
    uvicorn.run(app, host="0.0.0.0", port=8080)