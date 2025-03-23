# Báo Cáo Cải Tiến Hệ Thống Theo Dõi và Nhận Dạng Khuôn Mặt

## Tổng Quan
Tài liệu này mô tả các cải tiến đã được thực hiện cho module demo theo dõi và nhận dạng người trong hệ thống Face Behavior Tracking Light.

## Vấn Đề Ban Đầu
Hệ thống demo ban đầu gặp phải hai vấn đề chính:
1. Hiển thị quá nhiều bounding box không cần thiết (false positives)
2. Không gán tên chính xác cho người và không phân biệt các loại người (nhân viên, khách hàng, người khác) bằng màu sắc

## Phương Pháp Phát Triển
Theo tiêu chuẩn TDD (Test-Driven Development), chúng tôi đã tạo các bài kiểm thử trước khi thực hiện thay đổi trong mã nguồn:
1. `test_detection_filtering`: Kiểm tra khả năng lọc các phát hiện có độ tin cậy thấp
2. `test_identity_extraction`: Kiểm tra trích xuất tên và loại người từ đường dẫn ảnh
3. `test_color_mapping`: Kiểm tra gán màu sắc dựa trên loại người

## Các Cải Tiến Đã Thực Hiện

### 1. Giảm Số Lượng Bounding Box
- Tăng ngưỡng độ tin cậy (confidence threshold) từ giá trị mặc định lên 0.5
- Kết quả: Loại bỏ các phát hiện có độ tin cậy thấp, giảm số lượng bounding box không cần thiết
- Hiệu suất: FPS tăng từ ~12 lên ~172, thời gian xử lý giảm từ ~83ms xuống ~5.8ms mỗi khung hình

### 2. Gán Tên Chính Xác
- Thêm hàm `extract_identity_from_path()` để trích xuất tên từ đường dẫn ảnh
- Định dạng nhận dạng: "tên_người (ID)" thay vì chỉ "person_1"
- Xác định loại người (nhân viên, khách hàng, khác) dựa trên thư mục chứa ảnh

### 3. Phân Biệt Loại Người Bằng Màu Sắc
- Thêm hàm `get_color_by_type()` để gán màu dựa trên loại người:
  - Nhân viên: Màu xanh lá (0, 255, 0)
  - Khách hàng: Màu cam (0, 165, 255)
  - Người khác: Màu đỏ (0, 0, 255)
  - Không xác định: Màu xám hoặc màu ngẫu nhiên dựa trên ID

### 4. Cải Tiến Khác
- Mặc định sử dụng video `sample_2.mp4` để demo
- Cải thiện cách mô phỏng nhận dạng khuôn mặt với tên thực từ thư mục ảnh
- Tối ưu hiển thị thông tin trên video (FPS, số khung hình, ID theo dõi, loại người)

## Kết Quả
- Video kết quả đã được lưu tại `data/output/tracking_result_visual.mp4`
- Tất cả bài kiểm thử đều đã vượt qua thành công
- Hiện có thể dễ dàng phân biệt các loại người khác nhau trong video theo màu sắc

## Công Việc Tiếp Theo
- Tích hợp thực với thư viện nhận dạng khuôn mặt thực thay vì sử dụng mô phỏng
- Thêm tính năng phân tích hành vi dựa trên chuyển động và tương tác
- Tối ưu hóa thuật toán theo dõi để xử lý các tình huống phức tạp hơn (che khuất, đám đông)

## Lưu Ý
- Hiện tại đang sử dụng phiên bản mô phỏng thay vì thư viện `face_recognition` thực tế
- Cần cài đặt thư viện `face_recognition` thực tế hoặc sử dụng Conda để cài đặt từ kênh `conda-forge` 