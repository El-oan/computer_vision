import json

nb_path = "/Users/elo/Documents/coding/computer_vision/attention/attention.ipynb"

with open(nb_path, "r") as f:
    nb = json.load(f)

# 1. Fix Imports (Cell 1)
# The cell with imports is usually the second one (index 1), or the first code cell.
# Let's find the cell that has "import torch"
import_cell = None
for cell in nb["cells"]:
    if cell["cell_type"] == "code" and "import torch\n" in cell["source"]:
        import_cell = cell
        break

if import_cell:
    extra_imports = [
        "import torchtext.datasets as datasets\n",
        "from torchtext.data.utils import get_tokenizer\n",
        "from torchtext.vocab import build_vocab_from_iterator\n"
    ]
    # Check if already present to avoid duplicates
    existing_source = "".join(import_cell["source"])
    if "import torchtext.datasets" not in existing_source:
        import_cell["source"].extend(extra_imports)
        print("Added torchtext imports.")

# 2. Fix Broken Import (train_distributed_model)
# Find cell with "from the_annotated_transformer import train_worker"
broken_import_cell = None
for cell in nb["cells"]:
    if cell["cell_type"] == "code" and any("from the_annotated_transformer import train_worker" in line for line in cell["source"]):
        broken_import_cell = cell
        break

if broken_import_cell:
    new_source = []
    for line in broken_import_cell["source"]:
        if "from the_annotated_transformer import train_worker" in line:
            continue # Remove this line
        new_source.append(line)
    broken_import_cell["source"] = new_source
    print("Removed broken import.")

# 3. Add Main Block
# Check if main block already exists
has_main = False
if nb["cells"] and "if __name__ == \"__main__\":" in "".join(nb["cells"][-1]["source"]):
    has_main = True

if not has_main:
    main_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "\n",
            "if __name__ == \"__main__\":\n",
            "    # Load spacy models\n",
            "    spacy_de, spacy_en = load_tokenizers()\n",
            "    \n",
            "    # Build or load vocab\n",
            "    # Note: Multi30k might need internet access to download\n",
            "    vocab_src, vocab_tgt = load_vocab(spacy_de, spacy_en)\n",
            "    \n",
            "    config = {\n",
            "        \"batch_size\": 32,\n",
            "        \"distributed\": False,\n",
            "        \"num_epochs\": 2,\n",
            "        \"accum_iter\": 10,\n",
            "        \"base_lr\": 1.0,\n",
            "        \"max_padding\": 72,\n",
            "        \"warmup\": 3000,\n",
            "        \"file_prefix\": \"multi30k_model_\",\n",
            "    }\n",
            "    \n",
            "    train_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config)\n"
        ]
    }
    nb["cells"].append(main_cell)
    print("Added main execution block.")

with open(nb_path, "w") as f:
    json.dump(nb, f, indent=4)
