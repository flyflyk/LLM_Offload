## 環境需求

*   **硬體:**
    *   **NVIDIA GPU:** 一張支援 CUDA 的 NVIDIA 顯示卡。
    *   **RAM:** 取決於所選模型大小。
    *   **儲存空間:** 存放模型快取。
*   **軟體:**
    *   **OS:** 若要使用 `benchmark` 模式，建議使用 Linux 系統以獲得最準確的效能指標。
    *   **Python:** 建議版本 3.10+
    *   **Anaconda/Miniconda:** 用於管理 Python 環境和依賴。
    *   **CUDA Toolkit(建議 12.9):** [Cuda Toolkit 頁面](https://developer.nvidia.com/cuda-toolkit-archive)
    *   **PyTorch (GPU 版本):** [PyTorch 官方網站 Get Started 頁面](https://pytorch.org/get-started/locally/) 安裝與 CUDA 版本相容的 PyTorch。

## 安裝步驟

### 1. 取得專案檔案

```bash
git clone https://github.com/flyflyk/LLM_Offload.git
cd LLM_Offload
git submodule update --init --recursive # 初始化 submodule
```

### 2. 建立 Conda 環境並安裝依賴

```bash
# 建立 Conda 環境
conda create -n llm_inference_env python=3.12 -y
conda activate llm_inference_env

# 安裝依賴套件
pip install -r requirements.txt
pip install -e FlexLLMGen
```

## 使用方法

此專案的核心是 `main.py` 腳本，它提供了四種不同的執行模式，透過 `--mode` 參數進行切換。

---

### 模式 1: `autoflex` - 自動化策略推理 (推薦)

此模式會自動分析您的硬體配置，為指定的模型尋找最佳的 GPU/CPU/Disk 資源分配策略，並使用 FlexLLMGen 執行高效推理。

**執行指令:**

```bash
python main.py --mode autoflex [OPTIONS]
```

**[OPTIONS]:**

*   `--model`: 指定要使用的 Hugging Face 模型 (預設: `facebook/opt-1.3b`)。
*   `--input-len`: 輸入提示的長度 (token 數) (預設: `8`)。
*   `--gen-len`: 要生成的 token 數量 (預設: `32`)。
*   `--batch-size`: 一次處理的提示數量 (批次大小) (預設: `1`)。
*   `--path`: FlexLLMGen 模型權重的儲存路徑 (預設: `~/flexllmgen_cache`)。
*   `--offload-dir`: 權重卸載 (offload) 的暫存目錄 (預設: `~/flexllmgen_offload`)。
*   `--force-rerun-profiler`: 強制重新執行硬體分析，即使已有快取檔案。

**範例:**

```bash
# 使用 opt-2.7b 模型，並自動尋找最佳卸載策略進行推理
python main.py --mode autoflex --model facebook/opt-2.7b --input-len 512 --gen-len 64 --batch-size 4
```

---

### 模式 2: `accelerate` - 手動 Accelerate 推理

此模式直接使用 Hugging Face Accelerate 框架進行推理。您可以透過修改 `src/Accelerate/config.py` 來手動啟用或停用 KV Cache Offload 和 Streaming 模式。

**執行指令:**

```bash
python main.py --mode accelerate [OPTIONS]
```

**[OPTIONS]:**

*   `--model`: 指定要使用的 Hugging Face 模型 (預設: `facebook/opt-1.3b`)。
*   `--input-len`: 自動生成輸入提示的長度 (token 數) (預設: `8`)。
*   `--gen-len`: 要生成的 token 數量 (預設: `32`)。
*   `--batch-size`: 一次處理的提示數量 (批次大小) (預設: `1`)。

**`Accelerate/config.py` 設定:**

*   `ENABLE_OFFLOAD`: 是否啟用自動 offload 模式。
*   `ENABLE_KV_OFFLOAD`: 是否啟用 KV Cache Offload。
*   `OFFLOAD_FOLDER`: 權重 offload 的儲存路徑。

**範例:**

```bash
# 使用 Accelerate 進行標準推理
python main.py --mode accelerate --model facebook/opt-1.3b --input-len 128 --gen-len 128 --batch-size 2
```

#### 智慧型記憶體管理

`accelerate` 模式具備自動偵測並設定記憶體上限的功能。

*   **GPU VRAM**: 系統會自動偵測可用 VRAM，並使用其中 95% 作為模型加載的上限。
*   **CPU RAM**: 透過修改 `src/accelerate/config.py` 中的 `MAX_CPU_OFFLOAD` 參數來控制 CPU 的使用：
    *   `MAX_CPU_OFFLOAD = -1`: **(預設值)** 自動偵測可用的 RAM，並使用其中 95% 作為上限。
    *   `MAX_CPU_OFFLOAD = 0`: 不設限。

---

### 模式 3: `flexgen` - 手動 FlexGen 推理

此模式使用 FlexLLMGen 框架進行推理，也可以透過修改設定檔，手動調整權重在 GPU、CPU 之間的分配比例。

**執行指令:**

```bash
python main.py --mode flexgen [OPTIONS]
```

**[OPTIONS]:**

*   `--model`: 指定要使用的 Hugging Face 模型 (預設: `facebook/opt-1.3b`)。
*   `--input-len`: 輸入提示的長度 (token 數) (預設: `8`)。
*   `--gen-len`: 要生成的 token 數量 (預設: `32`)。
*   `--batch-size`: 一次處理的提示數量 (批次大小) (預設: `1`)。
*   `--log-file`: (可選) 將模型權重分佈的日誌儲存到指定檔案。

**範例:**

```bash
# 使用 FlexGen 進行推理，並將日誌存檔
python main.py --mode flexgen --model facebook/opt-1.3b --input-len 128 --gen-len 128 --batch-size 2
```

#### 手動設定卸載策略 (Manual Offloading Policy)

`flexgen` 模式的卸載策略可透過修改設定檔進行手動調整：

*   **設定檔路徑**: `src/flexgen/config.py`

---

### 模式 4: `benchmark` - 框架效能比較

此模式會對以下三種策略進行基準測試，並在最後提供一份詳細的效能比較報告：

1.  **Accelerate**: 使用 Hugging Face Accelerate (根據 `src/Accelerate/config.py` 的設定)。
2.  **FlexGen (All-GPU)**: 使用 FlexLLMGen 並將模型完全載入 GPU。
3.  **AutoFlex**: 使用 FlexLLMGen 並由系統自動尋找最佳卸載策略。

**執行指令:**

```bash
python main.py --mode benchmark [OPTIONS]
```

**[OPTIONS]:**

*   `--model`: 指定要測試的 Hugging Face 模型 (預設: `facebook/opt-1.3b`)。
*   `--batch-size`: 輸入的數量 (批次大小) (預設: `1`)。
*   `--input-len`: 輸入提示的長度 (token 數) (預設: `8`)。
*   `--gen-len`: 要生成的 token 數量 (預設: `32`)。
*   `--log-file`: (可選) 將所有框架的模型權重分佈日誌儲存到指定檔案。
*   `--force-rerun-profiler`: (可選) 在 AutoFlex 測試中，強制重新執行硬體分析。

**範例:**

```bash
# 比較三種策略在 opt-1.3b 模型上的表現
python main.py --mode benchmark --model facebook/opt-1.3b --input-len 64 --gen-len 64 --batch-size 4
```
