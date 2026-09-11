# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:** 2026-09-11

**Runtime Colab:** CPU

**Python / PyTorch / Ultralytics:** Python 3.x / PyTorch 2.x / Ultralytics 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`):
  - class_id: 468, class_name: "cab", rank: 1, score: 0.510915, taxonomy_name: "ImageNet-1K"
  
- Record này mô tả toàn ảnh như thế nào?
  - Record hạng 1 là kết quả dự đoán có score cao nhất của mô hình YOLO11 classification cho toàn bộ ảnh. Với ảnh traffic, model dự đoán lớp "cab" có class_id=468, rank=1, score=0.510915, nghĩa là model đánh giá "cab" (taxi) là chủ thể chiếm ưu thế nhất trong ảnh. Tương tự, với ảnh kitchen, model dự đoán "gong" (chiếc gong) với score=0.420192, và với ảnh dining, model dự đoán "restaurant" với score=0.792161.

- Ai định nghĩa class list mà checkpoint có thể dự đoán?
  - Class list được định nghĩa bởi taxonomy/dataset mà checkpoint được huấn luyện trên. Trong trường hợp này, taxonomy_name="ImageNet-1K", do đó các class như "cab", "gong", "restaurant", "dining_table", v.v. được hiểu theo hệ thống phân loại chuẩn của ImageNet-1K. Người dùng không tự định nghĩa lại class list khi thực hiện inference; model chỉ có thể dự đoán các lớp mà nó đã được huấn luyện trên.

- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?
  - Cần lưu cả class_id, class_name và taxonomy_name vì mỗi trường có một vai trò khác nhau:
    - class_id: mã định danh của lớp, dùng cho các hệ thống xử lý tự động.
    - class_name: tên lớp dễ đọc, giúp con người dễ hiểu kết quả.
    - taxonomy_name: cho biết lớp đó thuộc hệ thống phân loại nào. Cùng một class_id có thể có ý nghĩa khác nhau trong các taxonomy khác nhau. Việc lưu đầy đủ ba thông tin giúp kết quả không bị mơ hồ và có thể ánh xạ chính xác giữa model, class list và dữ liệu.

- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?
  - Nếu một ảnh có nhiều chủ thể cạnh tranh nhau, guideline cần xác định rõ:
    - Bài toán là single-label (một ảnh một nhãn duy nhất) hay multi-label (một ảnh nhiều nhãn)?
    - Nếu single-label, phải quy định tiêu chí để chọn chủ thể đại diện cho toàn ảnh (ví dụ: chủ thể chiếm diện tích lớn nhất, chiếm trung tâm ảnh, hoặc được đánh giá theo vai trò chính của ảnh).
    - Nếu multi-label, cần xác định khi nào nhiều lớp cùng được gán cho một ảnh (ví dụ: tất cả các chủ thể chính, hay chỉ những lớp chiếm tỷ lệ lớn hơn threshold nào đó).
    - Quy tắc này giúp việc tạo ground truth được nhất quán và loại trừ mơ hồ.

- Vì sao model score không phải ground truth?
  - Score của model không phải là ground truth vì:
    - Score chỉ thể hiện mức độ ưu tiên hoặc tin tưởng của model về một lớp trong quá trình dự đoán, không phải sự chắc chắn có thực.
    - Ví dụ score=0.510915 cho "cab" không có nghĩa ground truth chắc chắn là "cab", cũng không có nghĩa model đúng 51,09%.
    - Ground truth là nhãn chuẩn được xác định từ dữ liệu hoặc quá trình gán nhãn của con người, độc lập với dự đoán của model.
    - Muốn đánh giá model, cần so sánh prediction của model với ground truth độc lập. Nếu prediction trùng ground truth, gọi là TP (True Positive); nếu khác, gọi là FP (False Positive) hay FN (False Negative) tuỳ ngữ cảnh.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):
  - class_name: "person", score: 0.912625, bbox_xyxy: [385.33, 69.24, 498.92, 348.92], bbox_width: 113.58, bbox_height: 279.68

- Diễn giải vị trí box bằng lời:
  - Bounding box có tọa độ xyxy (x_min, y_min, x_max, y_max) = [385.33, 69.24, 498.92, 348.92] pixel. Cạnh trái của box ở x=385.33, cạnh trên ở y=69.24, cạnh phải ở x=498.92, cạnh dưới ở y=348.92. Box có chiều rộng 113.58 pixel và chiều cao 279.68 pixel, nằm ở vùng trung tâm-phải của ảnh kitchen (kích thước 640x427), bao quanh người đứng trong ảnh.

- So sánh số prediction ở hai threshold:
  - Tất cả các record detection sử dụng score_threshold=0.35. Không có dữ liệu với threshold khác được cung cấp. Kitchen có 11 đối tượng được dự đoán ở threshold 0.35. Nếu sử dụng threshold cao hơn (ví dụ 0.5), số lượng đối tượng dự đoán sẽ giảm vì chỉ những đối tượng có confidence cao hơn sẽ được giữ lại. Ngược lại, threshold thấp hơn (ví dụ 0.1) sẽ làm tăng số lượng nhưng cũng tăng false positive.

- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?
  - Threshold cao → độ bao phủ giảm (miss một số object), nhưng precision cao hơn (ít false alarm).
  - Threshold thấp → độ bao phủ cao hơn (catch được nhiều object), nhưng precision giảm (nhiều false positive).
  - Khối lượng reviewer tăng theo số lượng dự đoán: threshold thấp → reviewer phải xem và kiểm tra nhiều hơn, mệt hơn, dễ mắc lỗi; threshold cao → reviewer kiểm tra ít hơn nhưng có nguy cơ bỏ sót object thật.

- Đề xuất một quy tắc box chặt:
  - Box chặt = box sát sao bao quanh đối tượng, không có vùng trống hoặc phần background dư thừa.
  - Quy tắc: 
    - Không bao gồm hơn 10% vùng xung quanh không phải đối tượng (ví dụ background).
    - Không bỏ sót hơn 5% phần của đối tượng.
    - Nếu đối tượng có phần bị che khuất, box nên bao bên ngoài phần che khuất nếu có thể suy luận được; nếu không, cần escalation.

- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?
  - Nếu object bị che khuất một phần (ví dụ người đứng sau người khác):
    - Guideline nên quy định: vẫn tạo box bao toàn bộ phần nhìn thấy được, hay box nên bao phần bị che khuất dự đoán?
    - Hay là: nếu bị che khuất quá 30%, ghi chú và escalation cho annotator lead quyết định.
  - Nếu object bị cắt mép ảnh (chỉ nhìn thấy một phần):
    - Guideline: vẫn tạo box cho phần nhìn thấy được hay bỏ qua?
    - Nếu vẫn tạo box: phần cắt mép được xử lý như thế nào (box chỉ bao phần nhìn thấy hay theo biên ảnh)?

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):
  - instance_id: "kitchen-001", class_name: "person", score: 0.899318, polygon_point_count: 132, polygon_xy (5 điểm đầu): [[385.45, 66.44], [386.0, 65.0], [386.0, 64.0], [387.0, 63.0], ...]

- Polygon bổ sung chi tiết gì so với box?
  - Bounding box chỉ cung cấp một hình chữ nhật bao quanh đối tượng (4 tọa độ: x_min, y_min, x_max, y_max).
  - Polygon cung cấp danh sách chi tiết các điểm (vertices) theo thứ tự tạo thành đường biên/轮廓 của đối tượng, từ đó tái tạo hình dạng chính xác của đối tượng.
  - Ưu điểm của polygon: bao phủ chính xác hình dạng không đều (người, động vật, đồ vật lạ); loại bỏ phần background dư thừa trong box; chi tiết hơn cho các tác vụ yêu cầu độ chính xác cao.

- `instance_id` dùng để làm gì và không phải loại ID nào?
  - instance_id dùng để định danh duy nhất cho mỗi đối tượng (instance) riêng lẻ trong ảnh. Ví dụ "kitchen-001", "kitchen-002", v.v.
  - instance_id KHÔNG phải:
    - Class ID (class_id là định danh của lớp, chứ không phải của instance cụ thể; ví dụ tất cả "person" trong hình có class_id=0 nhưng instance_id khác).
    - Image ID (image/sample_id là định danh của ảnh, không phải instance).
    - Dùng làm ground truth score hay quality label; nó chỉ là mã theo dõi/traceability cho từng vật thể.

- Đề xuất một quy tắc biên mask:
  - Polygon/mask biên nên sát sao bao bề ngoài đối tượng.
  - Quy tắc:
    - Mỗi điểm polygon nên cách biên thực tế của đối tượng không quá ±1-2 pixel.
    - Không bao gồm background dư thừa: nếu nhìn thấy background rõ ràng bên trong polygon, cần điều chỉnh.
    - Với vùng mờ hoặc gradient: lấy điểm nằm ở biên giữa object và background.
    - Số điểm polygon nên cân bằng giữa chi tiết (nhiều điểm) và khả quản lý (không quá nhiều điểm gây overhead).

- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?
  - Vùng mờ (blurred boundary):
    - Guideline: nếu biên mờ hơn 3-5 pixel, annotator nên vẽ polygon theo "tâm" của vùng mờ hay "bên ngoài" bảo toàn đối tượng?
    - Hay escalation nếu mờ quá nhiều?
  - Tiếp xúc (two objects touching):
    - Guideline: nếu hai đối tượng cảm xúc, polygon nên vẽ thế nào để tách biệt rõ ràng?
    - Ví dụ: lấy giữa kẽ tiếp xúc, hay căn cứ vào hình dạng tự nhiên của từng object?
  - Che khuất (occlusion):
    - Guideline: nếu một phần object bị che bởi object khác, polygon nên bao bề ngoài của phần che khuất dự đoán hay chỉ phần nhìn thấy?
    - Nếu dự đoán phần che khuất, cần có training, hướng dẫn, hay escalation quyết định.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn lớp cho mỗi ảnh; format: image_id, class_id, class_name, taxonomy | Khi ảnh có nhiều chủ thể cạnh tranh (ví dụ: ảnh traffic, model dự đoán "cab" nhưng cũng có xe buýt). Guideline chưa rõ cách xử lý. | Annotator phải: (1) đọc guideline rõ tiêu chí chọn nhãn khi có nhiều lựa chọn; (2) kiểm tra hình ảnh cẩn thận, không chỉ tin vào label hoặc tiêu đề; (3) nếu có chủ thể ngang nhau, escalation cho reviewer lead. | Reviewer kiểm tra: (1) nhãn có phù hợp với nội dung ảnh không; (2) có bỏ sót hay confusion với các lớp tương tự không (ví dụ "gong" trong ảnh kitchen có thực tế hay model sai); (3) guideline được tuân thủ nhất quán không. |
| Phát hiện vật thể | Danh sách box (bbox) cho mỗi object; format: image_id, object_id, class_id, class_name, bbox_xyxy, score_threshold | Box không chặt (bao quá nhiều background); object bị che khuất hoặc cắt mép không được xử lý rõ (model dự đoán box có phần bị che khuất, guideline chưa định nghĩa rõ). Ví dụ ảnh kitchen: nếu người bị các đồ vật che khuất một phần, box nên bao toàn bộ dự đoán hay chỉ phần nhìn thấy. | Annotator phải: (1) vẽ box sát sao bao bề ngoài object; (2) nếu object bị che khuất >30%, kiểm tra guideline hay escalation; (3) không bỏ sót object nhỏ; (4) ghi chú nếu không chắc chắn. | Reviewer kiểm tra: (1) box chặt, không có dư thừa hay thiếu; (2) có bỏ sót object nào không (so sánh với hình minh họa); (3) các object cắt mép hay bị che khuất được xử lý nhất quán không; (4) số lượng box hợp lý (threshold 0.35). |
| Instance segmentation | Polygon/mask cho mỗi instance; format: image_id, instance_id, class_id, class_name, polygon_xy (danh sách điểm), score | Biên polygon không sát sao, bao phần background hay bỏ sót phần object; vùng mờ hoặc tiếp xúc giữa các object được xử lý không nhất quán. Ví dụ: polygon nhân vật trong kitchen quá rộng, bao phần nền; hoặc nếu có hai object tiếp xúc, polygon không tách biệt rõ. | Annotator phải: (1) vẽ polygon sát sao biên object; (2) mỗi điểm polygon cách biên thực tế ±1-2 pixel; (3) nếu biên mờ, escalation guideline; (4) khi object tiếp xúc, dùng tiêu chí nhất quán (ví dụ: lấy giữa kẽ); (5) ghi chú nếu có che khuất >50%. | Reviewer kiểm tra: (1) polygon khớp chính xác biên object; (2) không bao background dư thừa; (3) các instance tách biệt rõ ràng (đặc biệt tiếp xúc/che khuất); (4) số điểm polygon cân bằng (không quá ít, không quá nhiều); (5) consistency so với guideline. |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu:
  - Dữ liệu này là từ COCO 2017 validation set, đã công khai. Tuy nhiên, khi làm việc với dữ liệu thực từ dự án khách hàng:
    - Không tải dữ liệu chứa thông tin cá nhân (khuôn mặt, biển số xe, địa chỉ, v.v.) lên môi trường công khai.
    - Nếu dữ liệu chứa thông tin nhạy cảm, phải sử dụng encryption, anonymization, hoặc pseudonymization trước khi xử lý.
    - Chỉ chia sẻ dữ liệu với những người/hệ thống có quyền truy cập được phê duyệt.

- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho:
  - Người quản lý dự án (project lead / data manager).
  - Đội ngũ đạo đức/compliance nếu có liên quan đến dữ liệu nhạy cảm.
  - Không tự ý xóa hay sửa dữ liệu; ghi chú chi tiết vấn đề và chờ hướng dẫn.

## 6. Danh sách bằng chứng

- [x] `classification_predictions.json`
- [x] `detection_predictions.json`
- [x] `segmentation_predictions.json`
- [x] `IMAGE_ATTRIBUTION.md`
- [x] `visuals/classification_top5.png`
- [x] `visuals/detection_predictions.png`
- [x] `visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`.
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
