## 環境需求

*   **硬體:**
    *   **NVIDIA GPU:** 一張支援 CUDA 的 NVIDIA 顯示卡。
    *   **RAM:** 取決於所選模型大小。
    *   **儲存空間:** 存模型快取。
*   **軟體:**
    *   **OS:** 若要使用 `benchmark.py` ，需要 Linux 系統
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

2.  **初始化 Submodule:**
    使用 Git Submodule 來管理 FlexLLMGen 依賴。執行以下指令來下載：
    ```bash
    git submodule update --init --recursive
    ```

3.  **(推薦) 建立並啟用 Conda 環境:**
    ```bash
    conda create -n llm_inference_env python=3.12.6 -y
    conda activate llm_inference_env
    ```

4.  **安裝 PyTorch (GPU 版本):**
    *   前往 [PyTorch 官方網站 Get Started 頁面](https://pytorch.org/get-started/locally/)。

5.  **安裝其他依賴套件:**
    ```bash
    pip install -r requirements.txt
    ```

## 使用方法

`main.py` 是此專案的主要執行腳本，提供兩種操作模式：`inference` 和 `benchmark`。

### 1. 推理模式 (`inference`)

此模式會根據 `config.py` 中的設定，載入指定的模型並執行推理任務。

**步驟:**

1.  **設定 `config.py`:**
    *   開啟 `config.py` 檔案。
    *   根據您的需求修改參數，例如 `CHOSEN_MODEL`, `MAX_TOKENS`, `PROMPT_LIST` 等。
    *   您可以透過 `USE_ACCELERATE`, `ENABLE_STREAMING`, `ENABLE_KV_OFFLOAD` 等布林值來啟用或停用特定功能。

2.  **執行腳本:**
    ```bash
    python main.py --mode inference
    ```

### 2. 基準測試模式 (`benchmark`)

此模式會比較 **Accelerate** 和 **FlexLLMGen** 兩個框架的推理吞吐量，**Accelerate** 的行為會參照 `config.py` 的設定，而共用參數則由命令行傳入。

**執行指令:**

```bash
python main.py --mode benchmark [OPTIONS]
```

**可選參數 (OPTIONS):**

*   `--model`: 指定要測試的 Hugging Face 模型 (預設: `facebook/opt-1.3b`)。
*   `--input-nums`: 輸入的數量 (批次大小) (預設: `4`)。
*   `--input-len`: 輸入提示的長度 (token 數) (預設: `8`)。
*   `--gen-len`: 要生成的 token 數量 (預設: `32`)。
*   `--log-file`: (可選) 將模型權重分佈的日誌儲存到指定檔案。若未提供，則會直接輸出到控制台。

**範例:**

```bash
python main.py --mode benchmark --model facebook/opt-1.3b --input-nums 8 --input-len 128 --gen-len 64 --log-file log.log
```
