# LLM 推理框架

本專案提供一個統一的介面，用以執行和評測大型語言模型（LLM），並支援多種不同的卸載（offloading）與推理策略，它整合了 Hugging Face 的 `accelerate` 框架、 `FlexGen` 框架，分別用於標準及精細的的資源卸載。

## 環境需求

*   **硬體:**
    *   一張支援 CUDA 的 NVIDIA GPU。
    *   RAM 與儲存空間大小取決於您選擇的模型。
*   **軟體:**
    *   **作業系統:** Linux 系統(Ubuntu24.04)。
    *   **Anaconda/Miniconda:** 用於管理 Python 環境和依賴。
    *   **Python:** 3.10 以上(3.12.11)。
    *   **CUDA Toolkit:** 11.8 以上(12.9)。
    *   **PyTorch:** 與 CUDA 版本相容的 PyTorch。

## 安裝步驟

1.  **複製專案庫:**

    ```bash
    git clone https://github.com/flyflyk/LLM_Offload.git
    cd LLM_Offload
    git submodule update --init --recursive
    ```

2.  **建立 Conda 環境並安裝依賴套件:**

    ```bash
    # 建立並啟用 Conda 環境
    conda create -n llm_inference_env python=3.12 -y
    conda activate llm_inference_env

    # 安裝依賴套件
    pip install -r requirements.txt
    pip install -e FlexLLMGen
    ```

---

## 設定理念

將所有設定標準化為一條簡單的規則：

1.  **通用參數**：適用於所有模式的參數（如模型名稱、批次大小），統一透過**命令列參數**來設定。
2.  **模式專屬參數**：僅適用於特定推理模式的參數，則在 `src/configs/` 目錄下，對應的 **`<mode_name>.yaml` 設定檔**中進行配置。

---

## 使用方法

專案的主要啟動腳本是 `main.py`，可以透過 `--mode` 參數來選擇所需的推理策略。

### 通用命令列參數

適用於所有模式的參數：

*   `--mode`: 執行模式。可選：`accelerate`, `flexgen`, `autoflex`, `benchmark`。(預設: `accelerate`)
*   `--model`: 要使用的 Hugging Face 模型。 (預設: `facebook/opt-1.3b`)
*   `--input-len`: 輸入提示的 Token 長度。(預設: `50`)
*   `--gen-len`: 要生成的 Token 數量。(預設: `20`)
*   `--batch-size`: 一次處理的提示數量（批次大小）。(預設: `2`)
*   `--offload-dir`: 用於將張量（tensors）卸載到硬碟的通用目錄。(預設: `/mnt/ssd/offload_dir`)
*   `--log-file`: 日誌檔案的儲存路徑。(預設: `log.log`)

### 模式一: `accelerate`

使用標準的 Hugging Face `accelerate` 函式庫進行推理。

**模式專屬設定 (`src/configs/accelerate.yaml`):**

```yaml
enable_offload: true # 是否啟用權重卸載
enable_kv_offload: true # 是否啟用 KV 快取卸載
max_cpu_offload: -1 # 最大卸載到 RAM 的權重大小(-1: 代表無限制)
```

**範例:**

```bash
# 使用 accelerate 模式及批次大小 8 來執行 OPT-6.7B 模型
python main.py --mode accelerate --model facebook/opt-6.7b --batch-size 8
```

### 模式二: `flexgen`

使用 `FlexGen` 框架，允許手動控制權重、快取和激活值在 GPU、CPU 和硬碟之間的分配。

**模式專屬設定 (`src/configs/flexgen.yaml`):**

```yaml
path: "/mnt/ssd/offload_dir" # 用於儲存 FlexGen 下載和轉換後的模型權重的目錄
pin_weight: True # 是否使用鎖頁記憶體

# 權重卸載比例
w_gpu_percent: 100
w_cpu_percent: 0

# KV Cache 卸載比例
cache_gpu_percent: 100
cache_cpu_percent: 0

# Activations 卸載比例
act_gpu_percent: 100
act_cpu_percent: 0
```

**範例:**

```bash
# 執行一個大型模型
python main.py --mode flexgen --model facebook/opt-30b
```

### 模式三: `autoflex`

此模式會自動分析硬體，並為 `FlexGen` 尋找最佳的資源分配策略以最大化吞吐量（throughput）。

**模式專屬設定 (`src/configs/autoflex.yaml`):**

```yaml
force_rerun_profiler: False # 是否強制重新硬體分析，即便已存在分析快取
```

**範例:**

```bash
# 為 OPT-6.7B 自動尋找最佳策略
python main.py --mode autoflex --model facebook/opt-6.7b --input-len 512 --gen-len 64
```

### 模式四: `benchmark`

執行基準測試，比較 `accelerate`、`flexgen`（使用全 GPU 策略）和 `autoflex` 的效能。

**範例:**

```bash
# 使用批次大小 4，評測 OPT-2.7B 的表現
python main.py --mode benchmark --model facebook/opt-2.7b --batch-size 4
```
