## 環境需求

*   **硬體:**
    *   **NVIDIA GPU:** 一張支援 CUDA 的 NVIDIA 顯示卡。
    *   **RAM:** 取決於所選模型大小。
    *   **儲存空間:** 存模型快取。
*   **軟體:**
    *   **OS:** 若要使用 `benchmark.py` ，需要 Linux 系統
    *   **Python:** 建議版本 3.12.6+
    *   **Anaconda/Miniconda:** 用於管理 Python 環境和依賴。
    *   **NVIDIA 驅動程式:** 最新的穩定版 NVIDIA 顯示卡驅動程式。
    *   **Pytorch (GPU 版本):** 前往 [PyTorch 官方網站 Get Started 頁面](https://pytorch.org/get-started/locally/)。
    *   **CUDA Toolkit:** 與 PyTorch 版本相容的 CUDA Toolkit 版本。

## 安裝步驟

### 獲得檔案

```bash
git clone https://github.com/flyflyk/LLM_Offload.git
cd LLM_Offload
git submodule update --init --recursive # 初始化 submodule
```

### 套件環境

```bash
# Conda 環境
conda create -n llm_inference_env python=3.12.6 -y
conda activate llm_inference_env

# 下載依賴
pip install -r requirements.txt
pip install -e FlexLLMGen
```

## 使用方法

`main.py` 是主要執行腳本，提供兩種操作模式： `accelerate` 和 `benchmark`。

### Accelerate 設定檔 (`Accelerate/config.py`)

*   `ENABLE_STREAMING`: 是否啟用自動 offload 模式。
*   `ENABLE_KV_OFFLOAD`: 是否啟用 KV Cache Offload。
*   `OFFLOAD_FOLDER`: 權重 offload 的儲存路徑。
*   `OFFLOAD_FOLDER_MAX_CPU_OFFLOAD_RAM_GB`: Offload 到 CPU RAM 的最大限制。

---

### `accelerate` 模式

此模式僅使用 Accelerate 框架進行推理。

**執行指令:**

```bash
python main.py --mode accelerate [OPTIONS]
```

**[OPTIONS]:**

*   `--model`: 指定要使用的 Hugging Face 模型 (預設: `facebook/opt-1.3b`)。
*   `--input-len`: 自動生成輸入提示的長度 (token 數) (預設: `8`)。
*   `--gen-len`: 要生成的 token 數量 (預設: `32`)。
*   `--input-nums`: 一次處理的提示數量 (批次大小) (預設: `1`)。

**範例:**

```bash
# 使用預設的 input-len 進行推理
python main.py --mode accelerate

# 設定 input-len、生成長度和批次大小
python main.py --mode accelerate --model facebook/opt-1.3b --input-len 64 --gen-len 64 --input-nums 2
```
---

### `benchmark` 模式

此模式會比較 Accelerate 和 FlexLLMGen 兩個框架的推理吞吐量， Accelerate 的行為會參照 `config.py` 的設定，而共用參數則由命令行傳入。

**執行指令:**

```bash
python main.py --mode benchmark [OPTIONS]
```

**[OPTIONS]:**

*   `--model`: 指定要測試的 Hugging Face 模型 (預設: `facebook/opt-1.3b`)。
*   `--input-nums`: 輸入的數量 (批次大小) (預設: `1`)。
*   `--input-len`: 輸入提示的長度 (token 數) (預設: `8`)。
*   `--gen-len`: 要生成的 token 數量 (預設: `32`)。
*   `--log-file`: 將模型權重分佈的日誌儲存到指定檔案，若未提供，則會直接輸出到控制台。

**範例:**

```bash
python main.py --mode benchmark --model facebook/opt-1.3b --input-nums 4 --input-len 32 --gen-len 64 --log-file log.log
```
