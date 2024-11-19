# Tổng hợp một số extensions/tools mà nhóm sử dụng

- Xuất file `markdown preview` sang các định dạng khác: `Markdown Preview Enhanced`, đừng dùng `Markdown PDF` vì rất tệ.


# Hướng dẫn

1. Sử dụng `docker`

- Tải `Docker Desktop` cho [windows](https://docs.docker.com/desktop/setup/install/windows-install/)

- Hướng dẫn chi tiết [docker trên windows](https://www.youtube.com/watch?v=Gh1Sgknc6Fg&list=PLcgWZARZ3lBa4iKP8yudhVmovHJUEwSHs&index=1)

- Đã tạo `Dockerfile` cho mã nguồn, không cần chỉnh mã nguồn. [chi tiết](..\Dockerfile)

- Cách chạy dockerfile

    - Dùng `powershell` (hoặc `cmd`) trên `windows`

    - Truy cập đến thư mục của project (có chứa file `Dockerfile`)

    - Build `docker image` (cấu hình tương thích cho project trong `image`):

        > docker build -t graph-data-mining .

    - Sau khi build, run `docker container`:

        - chạy toàn bộ, không có tương tác với người dùng, sau khi hoàn thành `container` sẽ dừng hoạt động
            > docker run --name v1 graph-data-mining
        
        - chạy và tương tác với người dùng
            > docker run --name v1 -it graph-data-mining bash

    
- Kết quả được lưu trong thư mục mới `log` và `ovals`

    - Sao chép ra bên ngoài `docker containers`

        > docker cp <id>:<path_in_container> <path_on_host>

---

# Tiền xử lý mạng

- Cài đặt thuật toán tạo ma trận `X`, `C` và ma trận kề `A`

# Phân tích kết quả

- Kết quả trong các tập tin trong thư mục `ovals` là độ dị biệt của từng loại trên từng nút
- Rút trích các nút có độ dị biệt lớn và trực quan