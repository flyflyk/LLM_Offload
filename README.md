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

`main.py` 是主要執行腳本，提供兩種操作模式：`inference` 和 `benchmark`。

### 1. 設定檔 (`inference_engine/config.py`)

先設定 `inference_engine/config.py` 檔案。

*   `ENABLE_STREAMING`: 是否啟用串流輸出模式。
*   `ENABLE_KV_OFFLOAD`: 是否啟用 KV Cache Offload。
*   `OFFLOAD_FOLDER`: 權重 offload 的儲存路徑。
*   `OFFLOAD_FOLDER_MAX_CPU_OFFLOAD_RAM_GB`: Offload 到 CPU RAM 的最大限制。

### 2. 推理模式 (`inference`)

此模式僅使用 Accelerate 框架進行推理， prompt 將根據 `--input-len` 自動生成。

**執行指令:**

```bash
python main.py --mode inference --model [MODEL_NAME] [OPTIONS]
```

**必要參數:**

*   `--model`: 指定要使用的 Hugging Face 模型 (例如 `facebook/opt-1.3b`)。

**可選參數 (OPTIONS):**

*   `--input-len`: 輸入提示的長度 (token 數) (預設: `8`)。
*   `--gen-len`: 要生成的 token 數量 (預設: `32`)。
*   `--input-nums`: 一次處理的提示數量 (批次大小) (預設: `1`)。

**範例:**

```bash
# 使用預設的 input-len 進行推理
python main.py --mode inference --model facebook/opt-1.3b

# 設定 input-len、生成長度和批次大小
python main.py --mode inference --model facebook/opt-1.3b --input-len 64 --gen-len 64 --input-nums 2
```

### 3. 基準測試模式 (`benchmark`)

此模式會比較 **Accelerate** 和 **FlexLLMGen** 兩個框架的推理吞吐量，**Accelerate** 的行為會參照 `config.py` 的設定，而共用參數則由命令行傳入。

**執行指令:**

```bash
python main.py --mode benchmark --model [MODEL_NAME] [OPTIONS]
```

**必要參數:**

*   `--model`: 指定要測試的 Hugging Face 模型 (例如 `facebook/opt-1.3b`)。

**可選參數 (OPTIONS):**

*   `--input-nums`: 輸入的數量 (批次大小) (預設: `1`)。
*   `--input-len`: 輸入提示的長度 (token 數) (預設: `8`)。
*   `--gen-len`: 要生成的 token 數量 (預設: `32`)。
*   `--log-file`: 將模型權重分佈的日誌儲存到指定檔案，若未提供，則會直接輸出到控制台。

**範例:**

```bash
python main.py --mode benchmark --model facebook/opt-1.3b --input-nums 4 --input-len 32 --gen-len 64 --log-file log.log
```
