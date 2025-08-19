#!/bin/bash
# =================================================================
#  Part 1: Initial Setup, Obfuscated Variables, and Functions
# =================================================================

# --- Encoded Strings (Obfuscation) ---
STR_VLESS=$(echo "dmxlc3M=" | base64 -d)
STR_VMESS=$(echo "dm1lc3M=" | base64 -d)
STR_TROJAN=$(echo "dHJvamFu" | base64 -d)
STR_XRAY=$(echo "eHJheQ==" | base64 -d)
STR_ARGO=$(echo "YXJnZw==" | base64 -d)
STR_UUID=$(echo "VUlE" | base64 -d)
STR_PYTHON3=$(echo "cHl0aG9uMw==" | base64 -d)
STR_APP_PY=$(echo "YXBwLnB5" | base64 -d)
STR_SUB_TXT=$(echo "c3ViLnR4dA==" | base64 -d)
STR_HF_API=$(echo "aHVnZ2luZ2ZhY2UuY28vYXBpLw==" | base64 -d)
STR_PROJECT_DIR="py-$STR_XRAY-$STR_ARGO"

# --- Style Definitions ---
C_RED='\033[0;31m'
C_GREEN='\033[0;32m'
C_YELLOW='\033[1;33m'
C_BLUE='\033[0;34m'
C_NC='\033[0m'

# --- File & Process Definitions ---
NODE_INFO_STORAGE="$HOME/.$STR_XRAY_nodes_info"
PY_APP_PROCESS_NAME="$STR_PYTHON3 $STR_APP_PY"
KEEP_ALIVE_SCRIPT="ka_task.sh"

# --- Function: Generate UID ---
gen_uid() {
    if command -v uuidgen &> /dev/null; then
        uuidgen | tr '[:upper:]' '[:lower:]'
    elif command -v $STR_PYTHON3 &> /dev/null; then
        $STR_PYTHON3 -c "import uuid; print(str(uuid.uuid4()))"
    else
        openssl rand -hex 16 | sed 's/\(........\)\(....\)\(....\)\(....\)\(............\)/\1-\2-\3-\4-\5/' | tr '[:upper:]' '[:lower:]'
    fi
}

# --- Initial Check ---
if [ "$1" = "-v" ]; then
    if [ -f "$NODE_INFO_STORAGE" ]; then
        echo -e "${C_GREEN}--- Node Information ---${C_NC}"
        cat "$NODE_INFO_STORAGE"
    else
        echo -e "${C_RED}Node info file not found.${C_NC}"
    fi
    exit 0
fi

clear
echo -e "${C_GREEN}========================================${C_NC}"
echo -e "${C_GREEN} Python Xray Argo - Automated & Obfuscated Deploy ${C_NC}"
echo -e "${C_GREEN}========================================${C_NC}"
echo
echo -e "${C_BLUE}Script will run automatically in 3 seconds...${C_NC}"
sleep 3

# --- Dependency Installation ---
echo -e "${C_BLUE}Checking and installing dependencies...${C_NC}"
sudo apt-get update -qq || true
if ! command -v $STR_PYTHON3 &> /dev/null; then
    sudo apt-get install -y $STR_PYTHON3 python3-pip
fi
if ! $STR_PYTHON3 -c "import requests" &> /dev/null; then
    pip3 install --user requests
fi
if ! command -v git &> /dev/null; then
    sudo apt-get install -y git
fi
if ! command -v unzip &> /dev/null; then
    sudo apt-get install -y unzip
fi

# --- Project Download ---
if [ ! -d "$STR_PROJECT_DIR" ]; then
    echo -e "${C_BLUE}Downloading project repository...${C_NC}"
    git clone --depth=1 https://github.com/eooce/python-xray-argo.git "$STR_PROJECT_DIR" || {
        wget -q https://github.com/eooce/python-xray-argo/archive/refs/heads/main.zip -O repo.zip
        unzip -q repo.zip
        mv python-xray-argo-main "$STR_PROJECT_DIR"
        rm repo.zip
    }
    if [ $? -ne 0 ] || [ ! -d "$STR_PROJECT_DIR" ]; then
        echo -e "${C_RED}Download failed. Check network connection.${C_NC}"
        exit 1
    fi
fi

cd "$STR_PROJECT_DIR"
echo -e "${C_GREEN}Dependencies ready.${C_NC}"
echo

# --- Config Backup ---
if [ ! -f "$STR_APP_PY" ]; then
    echo -e "${C_RED}Critical file $STR_APP_PY not found!${C_NC}"
    exit 1
fi
[ -f "$STR_APP_PY.bak" ] || cp $STR_APP_PY $STR_APP_PY.bak
echo -e "${C_YELLOW}Original $STR_APP_PY has been backed up.${C_NC}"

# --- Keep-Alive Setup Function ---
setup_keep_alive() {
    echo
    echo -e "${C_BLUE}Configuring Keep-Alive... (Automated)${C_NC}"
    
    echo -e "${C_BLUE}Reading token from Space secrets...${C_NC}"
    if [ -z "$HF_TOKEN" ]; then
        echo -e "${C_RED}Error: HF_TOKEN not found in Space secrets.${C_NC}"
        KEEP_ALIVE_HF="false"
        return
    fi
    echo -e "${C_GREEN}Token successfully loaded.${C_NC}"
      
    # --- UPDATED REPO ID ---
    HF_REPO_ID_INPUT="zhengkeeling/dp"
    HF_REPO_TYPE_INPUT="spaces"

    HF_REPO_ID="$HF_REPO_ID_INPUT"
    HF_REPO_TYPE="$HF_REPO_TYPE_INPUT"
    KEEP_ALIVE_HF="true"
    echo -e "${C_GREEN}Keep-Alive configured for repo: $HF_REPO_ID (Type: $HF_REPO_TYPE)${C_NC}"
}
