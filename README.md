# Build-a-dataset-for-SOC

Khoá học / dự án xây dựng tập dữ liệu fine-tuning cho mô hình hỗ trợ SOC Tier 1 bằng cách chuyển các dữ liệu mạng từ CIC-IDS-2017 thành dạng instruction dữ liệu phù hợp cho huấn luyện hoặc đánh giá.

## Mục tiêu
- Tạo tập dữ liệu dạng chat instruction từ các mẫu cảnh báo mạng.
- Hỗ trợ mô hình hiểu ngữ cảnh SOC Tier 1 như: tóm tắt cảnh báo, phân loại sự kiện và đề xuất hành động tiếp theo.
- Tạo dữ liệu đầu vào cho quá trình fine-tuning hoặc thử nghiệm mô hình.

## Cấu trúc thư mục
- [chuyendoi.py](chuyendoi.py): chuyển dữ liệu từ file CSV preview sang định dạng JSONL theo cấu trúc chat.
- [preview.py](preview.py): pipeline xử lý dữ liệu CIC-IDS-2017, làm sạch, chuẩn hóa nhãn và xây dựng mẫu dữ liệu SOC.
- [dataset.jsonl](dataset.jsonl): tập dữ liệu ở định dạng JSONL đã chuyển đổi.
- [soc_dataset_350.jsonl](soc_dataset_350.jsonl): tập dữ liệu mẫu gồm 350 bản ghi đã được chọn lọc.
- [soc_dataset_preview.csv](soc_dataset_preview.csv): file CSV xem trước cho quá trình thử nghiệm.

## Quy trình chính
1. Chuẩn bị dữ liệu CIC-IDS-2017.
2. Làm sạch và chuẩn hóa nhãn tấn công.
3. Chọn mẫu đại diện cho các loại hoạt động như DoS, DDoS, PortScan, BruteForce, BENIGN, v.v.
4. Chuyển dữ liệu sang định dạng JSONL phù hợp cho fine-tuning.

## Yêu cầu môi trường
- Python 3.9+
- Các thư viện:
  - pandas
  - numpy

Có thể cài đặt bằng:

```bash
pip install pandas numpy
```

## Cách chạy
### 1. Chạy pipeline xây dựng dữ liệu
```bash
python preview.py
```

### 2. Chuyển dữ liệu preview sang JSONL
```bash
python chuyendoi.py
```

## Ghi chú
- File [preview.py](preview.py) hiện đang dùng đường dẫn dữ liệu cục bộ để đọc các file CSV CIC-IDS-2017. Bạn cần cập nhật đường dẫn trong biến `DATA_DIR` cho phù hợp với máy của mình.
- Repo này hiện tập trung vào việc chuẩn bị và chuyển đổi dữ liệu, chưa bao gồm bước huấn luyện mô hình hoặc xuất file GGUF.

## Mục đích sử dụng
Dự án này phù hợp để tạo dữ liệu huấn luyện cho các mô hình NLP/LLM trong lĩnh vực SOC, đặc biệt là các tác vụ phân tích cảnh báo và hỗ trợ phân loại sự cố.
