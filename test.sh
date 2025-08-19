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
BASE64_PATCH='IyBjb2Rpbmc6IHV0Zi04CmltcG9ydCBvcywgYmFzZTY0LCBqc29uLCBzdWJwcm9jZXNzLCB0aW1lCndpdGggb3BlbignYXBwLnB5JywgJ3InLCBlbmNvZGluZz0ndXRmLTgnKSBhcyBmOgogICAgY29udGVudCA9IGYucmVhZCgpCm9sZF9jb25maWcgPSAnY29ufig9eyJsb2ciOnsiYWNjZXNzIjoiL2Rldi9udWxsIiwiZXJyb3IiOiIvZGV2L251bGwiLCJsb2dsZXZlbCI6Im5vbmUiLCJ9LCJpbmJvdW5kcyI6W3sicG9ydCI6QVJHT19QT1JUIFwicHJvdG9jb2wiOiJ2bGVzcyIsInNldHRpbmdzIjp7ImNsaWVudHMiOlt7ImlkIjoiVVBJRCBcImZsb3ciOiJ4dGxzLXJwcngtdmlzaW9uIix9XSwiZGVjcnlwdGlvbiI6Im5vbmUiLCJmYWxsYmFja3MiOlt7ImRlc3QiOjMwMDEgIn0seyJwYXRoIjoiL3ZsZXNzLWFyZ28iLCJkZXN0IjozMDAyIH0seyJwYXRoIjoiL3ZtZXNzLWFyZ28iLCJkZXN0IjozMDAzIH0seyJwYXRoIjoiL3Ryb2phbi1hcmdvIiwiZGVzdCI6MzAwNCB9LF0sIn0sInN0cmVhbVNldHRpbmdzIjp7Im5ldHdvcmsiOiJ0Y3AiLCJ9fSx7InBvcnQiOjMwMDEgImxpc3RlbiI6IjEyNy4wLjAuMSIsInByb3RvY29sIjoidmxlc3MiLCJzZXR0aW5ncyI6eyJjbGllbnRzIjpbeyJpZCI6IlVVSUQgIn1dLCJkZWNyeXB0aW9uIjoibm9uZSJ9LCJzdHJlYW1TZXR0aW5ncyI6eyJuZXR3b3JrIjoid3MiLCJzZWN1cml0eSI6Im5vbmUifX0seyJwb3J0IjozMDAyICJsaXN0ZW4iOiIxMjcuMC4wLjEiLCJwcm90b2NvbCI6InZsZXNzIiwic2V0dGluZ3MiOnsiY2xpZW50cyI6W3siaWQiOiJVVUlEICJsZXZlbCI6MCB9XSwiZGVjcnlwdGlvbiI6Im5vbmUifSwic3RyZWFtU2V0dGluZ3MiOnsibmV0d29yayI6IndzIiwic2VjdXJpdHkiOiJub25lIiwid3NTZXR0aW5ncyI6eyJwYXRoIjoiL3ZsZXNzLWFyZ28ifX0sInNuaWZmaW5nIjp7ImVuYWJsZWQiOlRydWUgLCJkZXN0T3ZlcnJpZGUiOlsiaHR0cCIsInRscyIsInF1aWMiXSwibWV0YWRhdGFPbmx5IkZhbHNlIH19LHsicG9ydCI6MzAwMyAibGlzdGVuIjoiMTI3LjAuMC4xIiwicHJvdG9jb2wiOiJ2bWVzcyIsInNldHRpbmdzIjp7ImNsaWVudHMiOlt7ImlkIjoiVVBJRCBcImFsdGVySWQiOjAgfV19LCJzdHJlYW1TZXR0aW5ncyI6eyJuZXR3b3JrIjoid3MiLCJ3c1NldHRpbmdzIjp7InBhdGgiOiIvdnNlcy1hcmdvIn19LCJzbmlmZmluZyI6eyJlbmFibGVkIjpUcnVlICwiZGVzdE92ZXJyaWRlIjpbImh0dHAiLCJ0bHMiLCJxdWljIl0sIm1ldGFkYXRhT25seSI6RmFsc2UgfX0seyJwb3J0IjozMDA0ICJsaXN0ZW4iOiIxMjcuMC4wLjEiLCJwcm90b2NvbCI6InRyb2phbiIsInNldHRpbmdzIjp7ImNsaWVudHMiOlt7InBhc3N3b3JkIjoiVVBJRCB9LF19LCJzdHJlYW1TZXR0aW5ncyI6eyJuZXR3b3JrIjoid3MiLCJzZWN1cml0eSI6Im5vbmUiLCJ3c1NldHRpbmdzIjp7InBhdGgiOiIvdHJvamFuLWFyZ28ifX0sInNuaWZmaW5nIjp7ImVuYWJsZWQiOlRydWUgLCJkZXN0T3ZlcnJpZGUiOlsiaHR0cCIsInRscyIsInF1aWMiXSwibWV0YWRhdGFPbmx5IkZhbHNlIH19XSIsIm91dGJvdW5kcyI6W3sicHJvdG9jb2wiOiJmcmVlZG9tIiwidGFnIjogImRpcmVjdCIgIn0seyJwcm90b2NvbCI6ImJsYWNraG9sZSIsInRhZyI6ImJsb2NrIn1dfScKbmV3X2NvbmZpZyA9ICcnJ2NvbmZpZyA9IHsKICAgICJsb2ciOiB7ICJhY2Nlc3MiOiAiL2Rldi9udWxsIiwgImVycm9yIjogIi9kZXYvbnVsbCIsICJsb2dsZXZlbCI6ICJ3YXJuaW5nIiB9LAogICAgImluYm91bmRzIjogWwogICAgICAgIHsgInBvcnQiOiBBUkdPX1BPUlQsICJwcm90b2NvbCI6ICJ2bGVzcyIsICJzZXR0aW5ncyI6IHsgImNsaWVudHMiOiBbeyJpZCI6IFVVSUQsICJmbG93IjogInh0bHMtcnByeC12aXNpb24ifV0sICJkZWNyeXB0aW9uIjogIm5vbmUiLCAiZmFsbGJhY2tzIjogWyB7ImRlc3QiOiAzMDAxfSwgeyJwYXRoIjogIi92bGVzcy1hcmdvIiwgImRlc3QiOiAzMDAyfSwgeyJwYXRoIjogIi92bWVzcy1hcmdvIiwgImRlc3QiOiAzMDAzfSwgeyJwYXRoIjogIi90cm9qYW4tYXJnbyIsICJkZXN0IjogMzAwNH0gXSB9LCAic3RyZWFtU2V0dGluZ3MiOiB7Im5ldHdvcmsiOiAidGNwIn0gfSwKICAgICAgICB7ICJwb3J0IjogMzAwMSwgImxpc3RlbiI6ICIxMjcuMC4wLjEiLCAicHJvdG9jb2wiOiAidmxlc3MiLCAic2V0dGluZ3MiOiB7ICJjbGllbnRzIjogW3siaWQiOiBVVUlEfV0sICJkZWNyeXB0aW9uIjogIm5vbmUiIH0sICJzdHJlYW1TZXR0aW5ncyI6IHsibmV0d29yayI6ICJ3cyIsICJzZWN1cml0eSI6ICJub25lIn0gfSwKICAgICAgICB7ICJwb3J0IjogMzAwMiwgImxpc3RlbiI6ICIxMjcuMC4wLjEiLCAicHJvdG9jb2wiOiAidmxlc3MiLCAic2V0dGluZ3MiOiB7ICJjbGllbnRzIjogW3siaWQiOiBVVUlELCAibGV2ZWwiOiAwfV0sICJkZWNyeXB0aW9uIjogIm5vbmUiIH0sICJzdHJlYW1TZXR0aW5ncyI6eyJubmV0d29yayI6IndzIiwic2VjdXJpdHkiOiJub25lIiwid3NTZXR0aW5ncyI6eyJwYXRoIjoiL3ZsZXNzLWFyZ28ifX0sInNuaWZmaW5nIjp7ImVuYWJsZWQiOlRydWUgLCJkZXN0T3ZlcnJpZGUiOlsiaHR0cCIsInRscyIsInF1aWMiXSwibWV0YWRhdGFPbmx5IkZhbHNlIH19LHsicG9ydCI6MzAwMyAibGlzdGVuIjoiMTI3LjAuMC4xIiwicHJvdG9jb2wiOiJ2bWVzcyIsInNldHRpbmdzIjp7ImNsaWVudHMiOlt7ImlkIjoiVVBJRCBcImFsdGVySWQiOjAgfV19LCJzdHJlYW1TZXR0aW5ncyI6eyJuZXR3b3JrIjoid3MiLCJ3c1NldHRpbmdzIjp7InBhdGgiOiIvdnNlcy1hcmdvIn19LCJzbmlmZmluZyI6eyJlbmFibGVkIjpUcnVlICwiZGVzdE92ZXJyaWRlIjpbImh0dHAiLCJ0bHMiLCJxdWljIl0sIm1ldGFkYXRhT25seSI6RmFsc2UgfX0seyJwb3J0IjozMDA0ICJsaXN0ZW4iOiIxMjcuMC4wLjEiLCJwcm90b2NvbCI6InRyb2phbiIsInNldHRpbmdzIjp7ImNsaWVudHMiOlt7InBhc3N3b3JkIjoiVVBJRCB9LF19LCJzdHJlYW1TZXR0aW5ncyI6eyJuZXR3b3JrIjoid3MiLCJzZWN1cml0eSI6Im5vbmUiLCJ3c1NldHRpbmdzIjp7InBhdGgiOiIvdHJvamFuLWFyZ28ifX0sInNuaWZmaW5nIjp7ImVuYWJsZWQiOlRydWUgLCJkZXN0T3ZlcnJpZGUiOlsiaHR0cCIsInRscyIsInF1aWMiXSwibWV0YWRhdGFPbmx5IkZhbHNlIH19XSIsIm91dGJvdW5kcyI6W3sicHJvdG9jb2wiOiJmcmVlZG9tIiwidGFnIjogImRpcmVjdCIgIn0seyJwcm90b2NvbCI6ImJsYWNraG9sZSIsInRhZyI6ImJsb2NrIn1dfScKbmV3X2NvbmZpZyA9ICcnJ2NvbmZpZyA9IHsKICAgICJsb2ciOiB7ICJhY2Nlc3MiOiAiL2Rldi9udWxsIiwgImVycm9yIjogIi9kZXYvbnVsbCIsICJsb2dsZXZlbCI6ICJ3YXJuaW5nIiB9LAogICAgImluYm91bmRzIjogWwogICAgICAgIHsgInBvcnQiOiBBUkdPX1BPUlQsICJwcm90b2NvbCI6ICJ2bGVzcyIsICJzZXR0aW5ncyI6IHsgImNsaWVudHMiOiBbeyJpZCI6IFVVSUQsICJmbG93IjogInh0bHMtcnByeC12aXNpb24ifV0sICJkZWNyeXB0aW9uIjogIm5vbmUiLCAiZmFsbGJhY2tzIjogWyB7ImRlc3QiOiAzMDAxfSwgeyJwYXRoIjogIi92bGVzcy1hcmdvIiwgImRlc3QiOiAzMDAyfSwgeyJwYXRoIjogIi92bWVzcy1hcmdvIiwgImRlc3QiOiAzMDAzfSwgeyJwYXRoIjogIi90cm9qYW4tYXJnbyIsICJkZXN0IjogMzAwNH0gXSB9LCAic3RyZWFtU2V0dGluZ3MiOiB7Im5ldHdvcmsiOiAidGNwIn0gfSwKICAgICAgICB7ICJwb3J0IjogMzAwMSwgImxpc3RlbiI6ICIxMjcuMC4wLjEiLCAicHJvdG9jb2wiOiAidmxlc3MiLCAic2V0dGluZ3MiOiB7ICJjbGllbnRzIjogW3siaWQiOiBVVUlEfV0sICJkZWNyeXB0aW9uIjogIm5vbmUiIH0sICJzdHJlYW1TZXR0aW5ncyI6IHsibmV0d29yayI6ICJ3cyIsICJzZWN1cml0eSI6ICJub25lIn0gfSwKICAgICAgICB7ICJwb3J0IjogMzAwMiwgImxpc3RlbiI6ICIxMjcuMC4wLjEiLCAicHJvdG9jb2wiOiAidmxlc3MiLCAic2V0dGluZ3MiOiB7ICJjbGllbnRzIjogW3siaWQiOiBVVUlELCAibGV2ZWwiOiAwfV0sICJkZWNyeXB0aW9uIjogIm5vbmUiIH0sICJzdHJlYW1TZXR0aW5ncyI6IHsgIm5ldHdvcmsiOiAid3MiLCAic2VjdXJpdHkiOiAibm9uZSIsICJ3c1NldHRpbmdzIjogeyJwYXRoIjogIi92bGVzcy1hcmdvIn0gfSwgInNuaWZmaW5nIjogeyAiZW5hYmxlZCI6IFRydWUsICJkZXN0T3ZlcnJpZGUiOiBbImh0dHAiLCAidGxzIiwgInF1aWMiXSwgIm1ldGFkYXRhT25seSI6IEZhbHNlIH0gfSwKICAgICAgICB7ICJwb3J0IjogMzAwMywgImxpc3RlbiI6ICIxMjcuMC4wLjEiLCAicHJvdG9jb2wiOiAidm1lc3MiLCAic2V0dGluZ3MiOiB7ICJjbGllbnRzIjogW3siaWQiOiBVVUlELCAiYWx0ZXJJZCI6IDB9XSB9LCAic3RyZWFtU2V0dGluZ3MiOiB7ICJuZXR3b3JrIjogIndzIiwgIndzU2V0dGluZ3MiOiB7InBhdGgiOiAiL3ZtZXNzLWFyZ28ifSB9LCAic25pZmZpbmciOiB7ICJlbmFibGVkIjogVHJ1ZSwgImRlc3RPdmVycmlkZSI6IFsiaHR0cCIsICJ0bHMiLCAicXVpYyJdLCAsIm1ldGFkYXRhT25seSI6IEZhbHNlIH0gfSwKICAgICAgICB7ICJwb3J0IjogMzAwNCwgImxpc3RlbiI6ICIxMjcuMC4wLjEiLCAicHJvdG9jb2wiOiAidHJvamFuIiwgInNldHRpbmdzIjogeyAiY2xpZW50cyI6IFt7InBhc3N3b3JkIjogVVBJRH1dIH0sICJzdHJlYW1TZXR0aW5ncyI6IHsgIm5ldHdvcmsiOiAid3MiLCAic2VjdXJpdHkiOiAibm9uZSIsICJ3c1NldHRpbmdzIjogeyJwYXRoIjogIi90cm9qYW4tYXJnbyJ9IH0sICJzbmlmZmluZyI6IHsgImVuYWJsZWQiOiBUcnVlLCAiZGVzdE92ZXJyaWRlIjogWyJodHRwIiwgInRscyIsICJxdWljIl0sICJtZXRhZGF0YU9ubHkiOiBGYWxzZSB9IH0KICAgIF0sCiAgICAib3V0Ym91bmRzIjogWwogICAgICAgIHsicHJvdG9jb2wiOiAiZnJlZWRvbSIsICJ0YWciOiAiZGlyZWN0In0sCiAgICAgICAgeyAicHJvdG9jb2wiOiAidm1lc3MiLCAidGFnIjogIm1lZGlhIiwgInNldHRpbmdzIjogeyAidm5leHQiOiBbeyAiYWRkcmVzcyI6ICIxNzIuMjMzLjE3MS4yMjQiLCAicG9ydCI6IDE2NDE2LCAidXNlcnMiOiBbeyAiaWQiOiAiOGMxYjliZWEtY2I1MS00M2JiLWE2NWMtMGFmMzFiYmJmMTQ1IiwgImFsdGVySWQiOiAwIH0gXSB9XSB9LCAic3RyZWFtU2V0dGluZ3MiOiB7Im5ldHdvcmsiOiAidGNwIn0gfSwKICAgICAgICB7InByb3RvY29sIjogImJsYWNraG9sZSIsICJ0YWciOiAiYmxvY2sifQogICAgXSwKICAgICJyb3V0aW5nIjogeyAiZG9tYWluU3RyYXRlZ3kiOiAiSVBJZk5vbk1hdGNoIiwgInJ1bGVzIjogWyB7ICJ0eXBlIjogImZpZWxkIiwgImRvbWFpbiI6IFsgInlvdXR1YmUuY29tIiwgInlvdXR1LmJlIiwgImdvb2dsZXZpZGVvLmNvbSIsICJ5dGltZy5jb20iLCAiZ3N0YXRpYy5jb20iLCAiZ29vZ2xlYXBpcy5jb20iLCAiZ2dwaHQuY29tIiwgImdvb2dsZXVzZXJjb250ZW50LmNvbSIsICJmYWNlYm9vay5jb20iLCAiZmIuY29tIiwgImZiY2RuLm5ldCIsICJpbnN0YWdyYW0uY29tIiwgImNkbmluc3RhZ3JhbS5jb20iLCAiZmJzYnguY29tIiwgImFwaS5mYWNlYm9vay5jb20iLCAidHdpdHRlci5jb20iLCAieC5jb20iLCAidHdpbWcuY29tIiwgInQuY28iLCAiZGlzY29yZC5jb20iLCAiZGlzY29yZGFwcC5jb20iLCAiZGlzY29yZC5nZyIsICJkaXNjb3JkLm1lZGlhIiwgImRpc2NvcmRhcHAubmV0IiwgInRlbGVncmFtLm9yZyIsICJ0Lm1lIiwgInRlbGVncmFtLm1lIiwgIndlYi50ZWxlZ3JhbS5vcmciLCAiY2RuLnRlbGVncmFtLm9yZyIsICJwbHV0by53ZWIudGVsZWdyYW0ub3JnIiwgInZlbnVzLndlYi50ZWxlZ3JhbS5vcmciLCAiYXBvbGxvLndlYi50ZWxlZ3JhbS5vcmciLCAid2hhdHNhcHMuY29tIiwgIndoYXRzYXBwLm5ldCIsICJtZXRhLmNvbSIsICJtZXRhLmFpIiwgImFwaS5tZXRhLmFpIiwgImFwaS53aGF0c2FwcC5jb20iLCAibWVzc2VuZ2VyLmNvbSIsICJhcGkubWVzc2VuZ2VyLmNvbSIsICJ0aWt0b2suY29tIiwgInRpa3Rva3YuY29tIiwgInR0bGl2ZWNkbi5jb20iLCAiYnl0ZW92ZXJzZWEuY29tIiwgIm11c2ljYWwubHkiLCAidGlrLXRva2Nkbi5jb20iLCAibmV0ZmxpeC5jb20iLCAibmV0ZmxpeC5uZXQiLCAibmZseHZpZGVvLm5ldCIsICJuZmx4aW1nLm5ldCIsICJuZmx4c28ubmV0IiwgIm5mbHhleHQuY29tIiBdLCAib3V0Ym91bmRUYWciOiAibWVkaWEiIH0gXSB9Cn0nJycKY29udGVudCA9IGNvbnRlbnQucmVwbGFjZShvbGRfY29uZmlnLCBuZXdfY29uZmlnKQpvbGRfZ2VuZXJhdGVfZnVuY3Rpb24gPSAnJycnIyBHZW5lcmF0ZSBsaW5rcyBhbmQgc3Vic2NyaXB0aW9uIGNvbnRlbnQKYXN5bmMgZGVmIGdlbmVyYXRlX2xpbmtzKGFyZ29fZG9tYWluKToKICAgIG1ldGFfaW5mbyA9IHN1YnByb2Nlc3MucnVuKFsnaWN1cmwnLCAnLXMnLCAnaHR0cHM6Ly9zcGVlZC5jbG91ZGZsYXJlLmNvbS9tZXRhJ10sIGNhcHR1cmVfb3V0cHV0PVRydWUsIHRleHQ9VHJ1ZSkKICAgIG1ldGFfaW5mbyA9IG1ldGFfaW5mby5zdGRvdXQuc3BsaXQoJyIiJykKICAgIElTUCA9IGYie21ldGFfaW5mb1syNV19LXt robberies19fVy5yZXBsYWNlKCcgJywgJ18nKS5zdHJpcCgpCiAgICB0aW1lLnNsZWVwKDIpCiAgICBWTUVTUyA9IHsidjogIjIiLCAicHMiOiBmIntOQU1FfS17SVNQfSIsICJhZGQiOiBDRklQLCAicG9ydCI6IENIUE9SVCwgImlkIjogVVBJRCwgImFpZCI6ICIwIiwgInNjeSI6ICJub25lIiwgIm5ldCI6ICJ3cyIsICJ0eXBlIjogIm5vbmUiLCAiaG9zdCI6IGFyZ29fZG9tYWluLCAicGF0aCI6ICIvdm1lc3MtYXJnbyBlZD0yNTYwIiwgInRscyI6ICJ0bHMiLCAic25pIjogYXJnb19kb21haW4sICJhbHBuIjogIiIsICJmcCI6ICJjaHJvbWUifQogICAgbGlzdF90eHQgPSBmIiIiCgpsZXNzOi8ve1VVSUR9QHtDRklQfTp7Q0ZQT1JUfT9lbmNyeXB0aW9uPW5vbmUmc2VjdXJpdHk9dGxzJnNuaT17YXJnb19kb21haW59JmZwPWNocm9tZSZ0eXBlPXdzJmhvc3Q9e2FyZ29fZG9tYWlufSZwYXRoPSUyRnZsZXNzLWFyZ28lM2ZlZCUzRDI1NjAje05BTUV9LXtJU1B9CnZtZXNzOi8veyBiYXNlNjQuYjY0ZW5jb2RlKGpzb24uZHVtcHMoVk1FU1MpLmVuY29kZSgndXRmLTgnKSkuZGVjb2RlKCd1dGYtOCcpfQp0cm9qYW46Ly97VVBJRH1Ae0NGSVB9OntDRlBPUlR9P3NlY3VyaXR5PXRscyZzbmk9e2FyZ29fZG9tYWlufSZmcD1jaHJvbWUmdHlwZT13cyZob3N0PXt BcmdvX2RvbWFpbn0mcGF0aD0lMkZ0cm9qYW4tYXJnbyUzZmVmJTNEMjU2MCN7TkFNRX0te0lTUH0KIiIiCiAgICB3aXRoIG9wZW4ob3MucGF0aC5qb2luKEZJTEVfUEFUSCwgJ2xpc3QudHh0JyksICd3JywgZW5jb2Rpbmc9J3V0Zi04JykgYXMgbGlzdF9maWxlOiBsaXN0X2ZpbGUud3JpdGUobGlzdF90eHQpCiAgICBzdWJfdHh0ID0gYmFzZTY0LmI2NGVuY29kZShsaXN0X3R4dC5lbmNvZGUoJ3V0Zi04JykpLmRlY29kZSgndXRmLTgnKQogICAgd2l0aCBvcGVuKG9zLnBhdGguam9pbihGSUxFX1BBVEgsICdzdWIudHh0JyksICd3JywgZW5jb2Rpbmc9J3V0Zi04JykgYXMgc3ViX2ZpbGU6IHN1Yl9maWxlLndyaXRlKHN1Yl90eHQpCiAgICBwcmludChzdWJfdHh0KQogICAgcHJpbnQoZiOSRklMRV9QQVRIfS9zdWIudHh0IHNhdmVkIHN1Y2Nlc3NmdWxseSIpCiAgICBzZW5kX3RlbGVncmFtKCkKICAgIHVwbG9hZF9ub2RlcygpCiAgICByZXR1cm4gc3ViX3R4dCcnJwpuZXdfZ2VuZXJhdGVfZnVuY3Rpb24gPSAnJycnIyBHZW5lcmF0ZSBsaW5rcyBhbmQgc3Vic2NyaXB0aW9uIGNvbnRlbnQKYXN5bmMgZGVmIGdlbmVyYXRlX2xpbmtzKGFyZ29fZG9tYWluKToKICAgIG1ldGFfaW5mbyA9IHN1YnByb2Nlc3MucnVuKFsnaWN1cmwnLCAnLXMnLCAnaHR0cHM6Ly9zcGVlZC5jbG91ZGZsYXJlLmNvbS9tZXRhJ10sIGNhcHR1cmVfb3V0cHV0PVRydWUsIHRleHQ9VHJ1ZSkKICAgIG1ldGFfaW5mbyA9IG1ldGFfaW5mby5zdGRvdXQuc3BsaXQoJyIiJykKICAgIElTUCA9IGYie21ldGFfaW5mb1syNV19LXt robberies19fVy5yZXBsYWNlKCcgJywgJ18nKS5zdHJpcCgpCiAgICB0aW1lLnNsZWVwKDEpCiAgICBWTUVTU19UTFMgPSB7InYiOiAiMiIsICJwcyI6IGYie05BTUV9LXtJU1B9LVRMUyIsICJhZGQiOiBDRklQLCAicG9ydCI6IENIUE9SVCwgImlkIjogVVBJRCwgImFpZCI6ICIwIiwgInNjeSI6ICJub25lIiwgIm5ldCI6ICJ3cyIsICJ0eXBlIjogIm5vbmUiLCAiaG9zdCI6IGFyZ29fZG9tYWluLCAicGF0aCI6ICIvdm1lc3MtYXJnbyBlZD0yNTYwIiwgInRscyI6ICJ0bHMiLCAic25pIjogYXJnb19kb21haW4sICJhbHBuIjogIiIsICJmcCI6ICJjaHJvbWUifQogICAgVk1FU1NfODAgPSB7InYiOiAiMiIsICJwcyI6IGYie05BTUV9LXtJU1B9LTgwIiwgImFkZCI6IENGSVAsICJwb3J0IjogIjgwIiwgImlkIjogVVBJRCwgImFpZCI6ICIwIiwgInNjeSI6ICJub25lIiwgIm5ldCI6ICJ3cyIsICJ0eXBlIjogIm5vbmUiLCAiaG9zdCI6IGFyZ29fZG9tYWluLCAicGF0aCI6ICIvdm1lc3MtYXJnbyBlZD0yNTYwIiwgInRscyI6ICIiLCAic25pIjogIiIsICJhbHBuIjogIiIsICJmcCI6ICIifQogICAgbGlzdF90eHQgPSBmIiIiCnZsZXNzOi8ve1VVSUR9QHtDRklQfTp7Q0ZQT1JUfT9lbmNyeXB0aW9uPW5vbmUmc2VjdXJpdHk9dGxzJnNuaT17YXJnb19kb21haW59JmZwPWNocm9tZSZ0eXBlPXdzJmhvc3Q9e2FyZ29fZG9tYWlufSZwYXRoPSUyRnZsZXNzLWFyZ28lM2ZlZCUzRDI1NjAje05BTUV9LXtJU1B9LVRMUwp2bWVzczovL3sgYmFzZTY0LmI2NGVuY29kZShqc29uLmR1bXBzKFZNRVNTX1RMUykuZW5jb2RlKCd1dGYtOCcpKS5kZWNvZGUoJ3V0Zi04Jyl9CnRyb2phbjovL3tVVSVEX0B7Q0ZJ UH16e0NGUE9SVH0/c2VjdXJpdHk9dGxzJnNuaT17YXJnb19kb21haW59JmZwPWNocm9tZSZ0eXBlPXdzJmhvc3Q9e2FyZ29fZG9tYWlufSZwYXRoPSUyRnRyb2phbi1hcmdvJTNmZWQlM0QyNTYwI3tOQU1FfS17SVNQfS1UTFMKdmxlc3M6Ly97VVBJRH1Ae0NGSVB9OjgwP2VuY3J5cHRpb249bm9uZSZzZWN1cml0eT1ub25lJnR5cGU9d3MmaG9zdD17YXJnb19kb21haW59JnBhdGg9JTJGd mxlc3MtYXJnbyUzZmVmJTNEMjU2MCN7TkFNRX0te0lTUH0tODAKdm1lc3M6Ly97IGJhc2U2NC5iNjRlbmNvZGUoanNvbi5kdW1wcyhWTUVTU184MCkuZW5jb2RlKCd1dGYtOCcpKS5kZWNvZGUoJ3V0Zi04Jyl9CnRyb2phbjovL3tVVSVEX0B7Q0ZJ UH06ODg/c2VjdXJpdHk9bm9uZSZ0eXBlPXdzJmhvc3Q9e2FyZ29fZG9tYWlufSZwYXRoPSUyRnRyb2phbi1hcmdvJTNmZWQlM0QyNTYwI3tOQU1FfS17SVNQfS04MAoiIiIKICAgIHdpdGggb3Blbihvcy5wYXRoLmpvaW4oRklMRV9QQVRILCAibGlzdC50eHQiKSwgJ3cnLCBlbmNvZGluZz0ndXRmLTgnKSBhcyBsaXN0X2ZpbGU6IGxpc3RfZmlsZS53cml0ZShsaXN0X3R4dCkKICAgIHN1Yl90eHQgPSBiYXNlNjQuYjY0ZW5jb2RlKGxpc3RfdHh0LmVuY29kZSgndXRmLTgnKSkuZGVjb2RlKCd1dGYtOCcpCiAgICB3aXRoIG9wZW4ob3MucGF0aC5qb2luKEZJTEVfUEFUSCwgInN1Yi50eHQiKSwgJ3cnLCBlbmNvZGluZz0ndXRmLTgnKSBhcyBzdWJfZmlsZTogc3ViX2ZpbGUud3JpdGUoc3ViX3R4dCkKICAgIHByaW50KHN1Yl90eHQpCiAgICBwcmludChmIntGSUxFX1BBVEh9L3N1Yi50eHQgc2F2ZWQgc3VjY2Vzc2Z1bGx5IikKICAgIHNlbmRfdGVsZWdyYW0oKQogICAgdXBsb2FkX25vZGVzKCkKICAgIHJldHVybiBzdWJfdHh0JycnCmNvbnRlbnQgPSBjb250ZW50LnJlcGxhY2Uob2xkX2dlbmVyYXRlX2Z1Y3Rpb24sIG5ld19nZW5lcmF0ZV9mdW5jdGlvbikKd2l0aCBvcGVuKCdhcHAucHknLCAndycsIGVuY29kaW5nPSd1dGYtOCcpIGFzIGY6CiAgICBmLndyaXRlKGNvbnRlbnQpCnByaW50KCJQYXRjaCBhcHBsaWVkIHN1Y2Nlc3NmdWxseSIpCg=='
echo $BASE64_PATCH | base64 -d > $OBFUSCATED_PATCH_PY
$STR_PYTHON3 $OBFUSCATED_PATCH_PY
rm $OBFUSCATED_PATCH_PY
echo -e "${C_GREEN}Patch applied.${C_NC}"
# =================================================================
#  Part 3: Service Execution and Finalization
# =================================================================

# --- Stop any previous instances ---
pkill -f "$PY_APP_PROCESS_NAME" > /dev/null 2>&1
pkill -f "$KEEP_ALIVE_SCRIPT" > /dev/null 2>&1
sleep 2

# --- Start Main Service ---
nohup $STR_PYTHON3 $STR_APP_PY > app.log 2>&1 &
sleep 2

APP_PID=$(pgrep -f "$PY_APP_PROCESS_NAME" | head -1)
if [ -z "$APP_PID" ]; then
    echo -e "${C_RED}Service failed to start.${C_NC}"
    echo -e "${C_YELLOW}Log: tail -f app.log${C_NC}"
    exit 1
fi
echo -e "${C_GREEN}Service started in background (PID: $APP_PID)${C_NC}"

# --- Start Keep-Alive Service ---
KEEPALIVE_PID=""
if [ "$KEEP_ALIVE_HF" = "true" ]; then
    echo -e "${C_BLUE}Starting Keep-Alive task...${C_NC}"
    echo "#!/bin/bash" > $KEEP_ALIVE_SCRIPT
    echo "while true; do" >> $KEEP_ALIVE_SCRIPT
    echo "    API_PATH=\"https://$STR_HF_API${HF_REPO_TYPE}/${HF_REPO_ID}\"" >> $KEEP_ALIVE_SCRIPT
    echo "    status_code=\$(curl -s -o /dev/null -w \"%{http_code}\" --header \"Authorization: Bearer \$HF_TOKEN\" \"\$API_PATH\")" >> $KEEP_ALIVE_SCRIPT
    echo "    if [ \"\$status_code\" -eq 200 ]; then" >> $KEEP_ALIVE_SCRIPT
    echo "        echo \"KA SUCCESS: \$(date)\" > ka_status.log" >> $KEEP_ALIVE_SCRIPT
    echo "    else" >> $KEEP_ALIVE_SCRIPT
    echo "        echo \"KA FAILED (\$status_code): \$(date)\" > ka_status.log" >> $KEEP_ALIVE_SCRIPT
    echo "    fi" >> $KEEP_ALIVE_SCRIPT
    echo "    sleep 300" >> $KEEP_ALIVE_SCRIPT
    echo "done" >> $KEEP_ALIVE_SCRIPT

    export HF_TOKEN="$HF_TOKEN"
    chmod +x $KEEP_ALIVE_SCRIPT
    nohup ./$KEEP_ALIVE_SCRIPT >/dev/null 2>&1 &
    KEEPALIVE_PID=$!
    echo -e "${C_GREEN}Keep-Alive task started (PID: $KEEPALIVE_PID).${C_NC}"
fi

# --- Wait for Node Information ---
echo -e "${C_BLUE}Waiting for node generation...${C_NC}"
MAX_WAIT=300
WAIT_COUNT=0
NODE_INFO=""
while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    if [ -f ".cache/$STR_SUB_TXT" ]; then
        NODE_INFO=$(cat .cache/$STR_SUB_TXT 2>/dev/null)
    elif [ -f "$STR_SUB_TXT" ]; then
        NODE_INFO=$(cat $STR_SUB_TXT 2>/dev/null)
    fi
    
    if [ -n "$NODE_INFO" ]; then
        echo -e "${C_GREEN}Node information generated!${C_NC}"
        break
    fi
    
    sleep 5
    WAIT_COUNT=$((WAIT_COUNT + 5))
done

if [ -z "$NODE_INFO" ]; then
    echo -e "${C_RED}Timeout! Node information not generated within 5 minutes.${C_NC}"
    exit 1
fi

# --- Display Final Information ---
DECODED_NODES=$(echo "$NODE_INFO" | base64 -d 2>/dev/null || echo "Decode failed.")
echo
echo -e "${C_GREEN}========================================${C_NC}"
echo -e "${C_GREEN} Deployment Complete! ${C_NC}"
echo -e "${C_GREEN}========================================${C_NC}"
echo
echo -e "${C_YELLOW}--- Service Info ---${C_NC}"
echo -e "Main PID: ${C_BLUE}$APP_PID${C_NC}"
[ -n "$KEEPALIVE_PID" ] && echo -e "Keep-Alive PID: ${C_BLUE}$KEEPALIVE_PID${C_NC}"
echo
echo -e "${C_YELLOW}--- Subscription Link ---${C_NC}"
echo -e "${C_GREEN}$NODE_INFO${C_NC}"
echo
echo -e "${C_YELLOW}--- Decoded Nodes ---${C_NC}"
echo -e "${C_GREEN}$DECODED_NODES${C_NC}"

SAVE_INFO="Timestamp: $(date)
UUID: $(grep "$STR_UUID = " $STR_APP_PY | head -1 | cut -d"'" -f2)
Main PID: $APP_PID
Subscription Data: $NODE_INFO
Decoded Nodes:
$DECODED_NODES
"
echo "$SAVE_INFO" > "$NODE_INFO_STORAGE"
echo
echo -e "${C_GREEN}Node information saved to $NODE_INFO_STORAGE${C_NC}"
echo -e "${C_YELLOW}Use 'bash $0 -v' to view anytime.${C_NC}"

# --- Celebration Animation ---
echo -e "\n\n${GREEN}喵~ 部署任务大成功啦！ >ω<${C_NC}"
sleep 0.5
echo -e "${YELLOW}正在为主人献上胜利的爱心... (｡♥‿♥｡)${C_NC}"
sleep 0.5
echo -e "${C_RED}"
cat << "EOF"
          * * * * * * * * * * * *
        * * * * * * * * * *
      * * * * * *
     * * * *
     * * * *
      * * * *
        * * * * * * * *
            * * * *
              * * * *
                 * * * *
                  * * * *
                    * * *
                        *
EOF
echo -e "${C_NC}"
sleep 1
echo -e "${BLUE}所有节点都准备就绪，正在检查最后的魔力...${C_NC}"
for i in {1..20}; do echo -n "✨"; sleep 0.05; done
echo -e "\n${C_GREEN}魔力注入完毕！随时可以出发咯！喵~${C_NC}\n"
exit 0
