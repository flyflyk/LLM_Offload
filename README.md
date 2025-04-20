## 環境需求

*   **作業系統:** Windows 10/11。
*   **硬體:**
    *   **NVIDIA GPU:** 一張支援 CUDA 的 NVIDIA 顯示卡。
    *   **RAM:** 取決於所選模型大小
    *   **儲存空間:** 存放 Python 環境、函式庫以及下載的模型快取。
*   **軟體:**
    *   **Python:** 建議版本 3.12.6+
    *   **Anaconda:** 用於管理 Python 環境和依賴。
    *   **NVIDIA 驅動程式:** 最新的穩定版 NVIDIA 顯示卡驅動程式。
    *   **CUDA Toolkit:** 與 PyTorch 版本相容的 CUDA Toolkit 版本。

## 安裝步驟

1.  **Clone 專案庫:**
    ```bash
    git clone https://github.com/flyflyk/LLM_Offload.git
    cd LLM_Offload
    ```

2.  **(推薦) 建立並啟用 Conda 環境:**
    ```bash
    conda create -n llm_inference_env python=3.12.6 -y
    conda activate llm_inference_env
    ```

3.  **安裝 PyTorch (GPU 版本):**
    *   前往 [PyTorch 官方網站 Get Started 頁面](https://pytorch.org/get-started/locally/)。

4.  **安裝其他依賴套件:**
    ```bash
    pip install -r requirements.txt
    ```

## 使用方法

1.  **設定模型:**
    在 `main.py` (或其他設定檔) 中，修改 `CHOSEN_MODEL` 變數來選擇您想使用的 Hugging Face 模型 (例如 `"gpt2-medium"`, `"facebook/opt-1.3b"`)。

2.  **執行主腳本:**
    ```bash
    python main.py
    ```