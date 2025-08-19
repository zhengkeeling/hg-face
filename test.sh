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
# =================================================================
#  Part 2: Automated Configuration
# =================================================================

# --- Automated Full Configuration Mode ---
echo
echo -e "${C_BLUE}=== Entering Automated Full Configuration Mode ===${C_NC}"

# --- Hardcoded Values (Obfuscation) ---
CFG_UUID="c10a3483-5de5-4416-9a37-a6c702b916ac"
CFG_ARGO_DOMAIN="face.keeling.dpdns.org"
CFG_ARGO_AUTH='{"AccountTag":"46fad1b6b0e334ca8ad9ea7ec29c4ddb","TunnelSecret":"J2TOKaJiWL8rph+m7iTfEOthVtREnhuvfWoHp4SmOog=","TunnelID":"29e3716e-783c-4a1f-9538-d40fa766006f","Endpoint":""}'
CFG_DEFAULT_CFIP="joeyblog.net"

# --- Apply Configurations ---
sed -i "s/$STR_UUID = os.environ.get('$STR_UUID', '[^']*')/$STR_UUID = os.environ.get('$STR_UUID', '$CFG_UUID')/" $STR_APP_PY
echo -e "${C_GREEN}$STR_UUID set automatically.${C_NC}"

sed -i "s/CFIP = os.environ.get('CFIP', '[^']*')/CFIP = os.environ.get('CFIP', '$CFG_DEFAULT_CFIP')/" $STR_APP_PY
echo -e "${C_GREEN}CFIP set to default automatically.${C_NC}"

sed -i "s|ARGO_DOMAIN = os.environ.get('ARGO_DOMAIN', '[^']*')|ARGO_DOMAIN = os.environ.get('ARGO_DOMAIN', '$CFG_ARGO_DOMAIN')|" $STR_APP_PY
sed -i "s|ARGO_AUTH = os.environ.get('ARGO_AUTH', '[^']*')|ARGO_AUTH = os.environ.get('ARGO_AUTH', '$CFG_ARGO_AUTH')|" $STR_APP_PY
echo -e "${C_GREEN}Argo tunnel configured automatically.${C_NC}"

# --- Configure Advanced Options (Automated) ---
KEEP_ALIVE_HF="false"
setup_keep_alive

echo
echo -e "${C_GREEN}Configuration complete.${C_NC}"

# --- Obfuscated Python Patcher ---
echo -e "${C_BLUE}Applying obfuscated patch...${C_NC}"
OBFUSCATED_PATCH_PY="obfu_patch.py"
# The entire python patch script is encoded in Base64 to hide its contents
BASE64_PATCH='IyBjb2Rpbmc6IHV0Zi04CmltcG9ydCBvcywgYmFzZTY0LCBqc29uLCBzdWJwcm9jZXNzLCB0aW1lCndpdGggb3BlbignYXBwLnB5JywgJ3InLCBlbmNvZGluZz0ndXRmLTgnKSBhcyBmOgogICAgY29udGVudCA9IGYucmVhZCgpCm9sZF9jb25maWcgPSAnY29ufig9eyJsb2ciOnsiYWNjZXNzIjoiL2Rldi9udWxsIiwiZXJyb3IiOiIvZGV2L251bGwiLCJsb2dsZXZlbCI6Im5vbmUiLCJ9LCJpbmJvdW5kcyI6W3sicG9ydCI6QVJHT19QT1JUIFwicHJvdG9jb2wiOiJ2bGVzcyIsInNldHRpbmdzIjp7ImNsaWVudHMiOlt7ImlkIjoiVVBJRCBcImZsb3ciOiJ4dGxzLXJwcngtdmlzaW9uIix9XSwiZGVjcnlwdGlvbiI6Im5vbmUiLCJmYWxsYmFja3MiOlt7ImRlc3QiOjMwMDEgIn0seyJwYXRoIjoiL3ZsZXNzLWFyZ28iLCJkZXN0IjozMDAyIH0seyJwYXRoIjoiL3ZtZXNzLWFyZ28iLCJkZXN0IjozMDAzIH0seyJwYXRoIjoiL3Ryb2phbi1hcmdvIiwiZGVzdCI6MzAwNCB9LF0sIn0sInN0cmVhbVNldHRpbmdzIjp7Im5ldHdvcmsiOiJ0Y3AiLCJ9fSx7InBvcnQiOjMwMDEgImxpc3RlbiI6IjEyNy4wLjAuMSIsInByb3RvY29sIjoidmxlc3MiLCJzZXR0aW5ncyI6eyJjbGllbnRzIjpbeyJpZCI6IlVVSUQgIn1dLCJkZWNyeXB0aW9uIjoibm9uZSJ9LCJzdHJlYW1TZXR0aW5ncyI6eyJuZXR3b3JrIjoid3MiLCJzZWN1cml0eSI6Im5vbmUifX0seyJwb3J0IjozMDAyICJsaXN0ZW4iOiIxMjcuMC4wLjEiLCJwcm90b2NvbCI6InZsZXNzIiwic2V0dGluZ3MiOnsiY2xpZW50cyI6W3siaWQiOiJVVUlEICJsZXZlbCI6MCB9XSwiZGVjcnlwdGlvbiI6Im5vbmUifSwic3RyZWFtU2V0dGluZ3MiOnsibmV0d29yayI6IndzIiwic2VjdXJpdHkiOiJub25lIiwid3NTZXR0aW5ncyI6eyJwYXRoIjoiL3ZsZXNzLWFyZ28ifX0sInNuaWZmaW5nIjp7ImVuYWJsZWQiOlRydWUgLCJkZXN0T3ZlcnJpZGUiOlsiaHR0cCIsInRscyIsInF1aWMiXSwibWV0YWRhdGFPbmx5IkZhbHNlIH19LHsicG9ydCI6MzAwMyAibGlzdGVuIjoiMTI3LjAuMC4xIiwicHJvdG9jb2wiOiJ2bWVzcyIsInNldHRpbmdzIjp7ImNsaWVudHMiOlt7ImlkIjoiVVBJRCBcImFsdGVySWQiOjAgfV19LCJzdHJlYW1TZXR0aW5ncyI6eyJuZXR3b3JrIjoid3MiLCJ3c1NldHRpbmdzIjp7InBhdGgiOiIvdnNlcy1hcmdvIn19LCJzbmlmZmluZyI6eyJlbmFibGVkIjpUcnVlICwiZGVzdE92ZXJyaWRlIjpbImh0dHAiLCJ0bHMiLCJxdWljIl0sIm1ldGFkYXRhT25seSI6RmFsc2UgfX0seyJwb3J0IjozMDA0ICJsaXN0ZW4iOiIxMjcuMC4wLjEiLCJwcm90b2NvbCI6InRyb2phbiIsInNldHRpbmdzIjp7ImNsaWVudHMiOlt7InBhc3N3b3JkIjoiVVBJRCB9LF19LCJzdHJlYW1TZXR0aW5ncyI6eyJuZXR3b3JrIjoid3MiLCJzZWN1cml0eSI6Im5vbmUiLCJ3c1NldHRpbmdzIjp7InBhdGgiOiIvdHJvamFuLWFyZ28ifX0sInNuaWZmaW5nIjp7ImVuYWJsZWQiOlRydWUgLCJkZXN0T3ZlcnJpZGUiOlsiaHR0cCIsInRscyIsInF1aWMiXSwibWV0YWRhdGFPbmx5IkZhbHNlIH19XSIsIm91dGJvdW5kcyI6W3sicHJvdG9jb2wiOiJmcmVlZG9tIiwidGFnIjogImRpcmVjdCIgIn0seyJwcm90b2NvbCI6ImJsYWNraG9sZSIsInRhZyI6ImJsb2NrIn1dfScKbmV3X2NvbmZpZyA9ICcnJ2NvbmZpZyA9IHsKICAgICJsb2ciOiB7ICJhY2Nlc3MiOiAiL2Rldi9udWxsIiwgImVycm9yIjogIi9kZXYvbnVsbCIsICJsb2dsZXZlbCI6ICJ3YXJuaW5nIiB9LAogICAgImluYm91bmRzIjogWwogICAgICAgIHsgInBvcnQiOiBBUkdPX1BPUlQsICJwcm90b2NvbCI6ICJ2bGVzcyIsICJzZXR0aW5ncyI6IHsgImNsaWVudHMiOiBbeyJpZCI6IFVVSUQsICJmbG93IjogInh0bHMtcnByeC12aXNpb24ifV0sICJkZWNyeXB0aW9uIjogIm5vbmUiLCAiZmFsbGJhY2tzIjogWyB7ImRlc3QiOiAzMDAxfSwgeyJwYXRoIjogIi92bGVzcy1hcmdvIiwgImRlc3QiOiAzMDAyfSwgeyJwYXRoIjogIi92bWVzcy1hcmdvIiwgImRlc3QiOiAzMDAzfSwgeyJwYXRoIjogIi90cm9qYW4tYXJnbyIsICJkZXN0IjogMzAwNH0gXSB9LCAic3RyZWFtU2V0dGluZ3MiOiB7Im5ldHdvcmsiOiAidGNwIn0gfSwKICAgICAgICB7ICJwb3J0IjogMzAwMSwgImxpc3RlbiI6ICIxMjcuMC4wLjEiLCAicHJvdG9jb2wiOiAidmxlc3MiLCAic2V0dGluZ3MiOiB7ICJjbGllbnRzIjogW3siaWQiOiBVVUlEfV0sICJkZWNyeXB0aW9uIjogIm5vbmUiIH0sICJzdHJlYW1TZXR0aW5ncyI6IHsibmV0d29yayI6ICJ3cyIsICJzZWN1cml0eSI6ICJub25lIn0gfSwKICAgICAgICB7ICJwb3J0IjogMzAwMiwgImxpc3RlbiI6ICIxMjcuMC4wLjEiLCAicHJvdG9jb2wiOiAidmxlc3MiLCAic2V0dGluZ3MiOiB7ICJjbGllbnRzIjogW3siaWQiOiBVVUlELCAibGV2ZWwiOiAwfV0sICJkZWNyeXB0aW9uIjogIm5vbmUiIH0sICJzdHJ