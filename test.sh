#!/bin/bash
# =================================================================
#  FINAL DIAGNOSTIC VERSION - With Intelligent Health Check
# =================================================================

# --- Style Definitions ---
C_RED='\033[0;31m'; C_GREEN='\033[0;32m'; C_YELLOW='\033[1;33m'; C_BLUE='\033[0;34m'; C_NC='\033[0m'

# --- 1. Password Verification ---
SECRET_PASS="keeling"
echo -e -n "${C_YELLOW}Please enter the password to continue: ${C_NC}"
read -s USER_INPUT
echo
if [ "$USER_INPUT" != "$SECRET_PASS" ]; then
    echo -e "${C_RED}Authentication failed. Exiting.${C_NC}"; exit 1
fi
echo -e "${C_GREEN}Authentication successful.${C_NC}"; echo

# --- Light Obfuscation ---
S_PYTHON3=$(echo "cHl0aG9uMw==" | base64 -d)
S_APP_PY=$(echo "YXBwLnB5" | base64 -d)
PROJECT_DIR_NAME="python-xray-argo"
NODE_INFO_FILE="$HOME/.xray_nodes_info"

# --- Initial Check ---
if [ "$1" = "-v" ]; then
    if [ -f "$NODE_INFO_FILE" ]; then
        cat "$NODE_INFO_FILE"
    else
        echo -e "${C_RED}Node info file not found.${C_NC}"
    fi
    exit 0
fi

clear
echo -e "${C_GREEN}========================================${NC}"
echo -e "${C_GREEN} Python Xray Argo - Automated & Restored ${NC}"
echo -e "${C_GREEN}========================================${NC}"
echo
echo -e "${C_BLUE}Based on: https://github.com/eooce/python-xray-argo${C_NC}"
echo -e "${C_GREEN}Script will run automatically...${C_NC}"; sleep 2

# --- Dependency Installation ---
echo -e "${C_BLUE}Checking and installing dependencies...${C_NC}"
sudo apt-get update -qq || true
if ! command -v $S_PYTHON3 &> /dev/null; then
    sudo apt-get install -y python3 python3-pip || true
fi
if ! $S_PYTHON3 -c "import requests" &> /dev/null; then
    pip3 install --user requests
fi
if ! command -v git &> /dev/null; then
    sudo apt-get install -y git || true
fi
if ! command -v unzip &> /dev/null; then
    sudo apt-get install -y unzip || true
fi

# --- Project Download ---
if [ ! -d "$PROJECT_DIR_NAME" ]; then
    echo -e "${C_BLUE}Downloading project repository...${C_NC}"
    git clone --depth=1 https://github.com/eooce/python-xray-argo.git "$PROJECT_DIR_NAME" || {
        wget -q https://github.com/eooce/python-xray-argo/archive/refs/heads/main.zip -O repo.zip
        unzip -q repo.zip
        mv python-xray-argo-main "$PROJECT_DIR_NAME"
        rm repo.zip
    }
    if [ $? -ne 0 ] || [ ! -d "$PROJECT_DIR_NAME" ]; then
        echo -e "${C_RED}Download failed. Check network connection.${C_NC}"; exit 1
    fi
fi

cd "$PROJECT_DIR_NAME"
echo -e "${C_GREEN}Dependencies ready.${C_NC}"; echo

# --- Config Backup & Setup ---
if [ ! -f "$S_APP_PY" ]; then
    echo -e "${C_RED}Critical file $S_APP_PY not found!${C_NC}"; exit 1
fi
[ -f "$S_APP_PY.bak" ] || cp $S_APP_PY $S_APP_PY.bak
echo -e "${C_YELLOW}Original $S_APP_PY has been backed up.${C_NC}"

# --- Automated Configuration (Non-interactive) ---
echo -e "${C_BLUE}=== Entering Automated Full Configuration ===${C_NC}"

UUID_INPUT="c10a3483-5de5-4416-9a37-a6c702b916ac"
sed -i "s/UUID = os.environ.get('UUID', '[^']*')/UUID = os.environ.get('UUID', '$UUID_INPUT')/" $S_APP_PY

ARGO_DOMAIN_INPUT="face.keeling.dpdns.org"
sed -i "s|ARGO_DOMAIN = os.environ.get('ARGO_DOMAIN', '[^']*')|ARGO_DOMAIN = os.environ.get('ARGO_DOMAIN', '$ARGO_DOMAIN_INPUT')|" $S_APP_PY

ARGO_AUTH_INPUT='{"AccountTag":"46fad1b6b0e334ca8ad9ea7ec29c4ddb","TunnelSecret":"J2TOKaJiWL8rph+m7iTfEOthVtREnhuvfWoHp4SmOog=","TunnelID":"29e3716e-783c-4a1f-9538-d40fa766006f","Endpoint":""}'
sed -i "s|ARGO_AUTH = os.environ.get('ARGO_AUTH', '[^']*')|ARGO_AUTH = os.environ.get('ARGO_AUTH', '$ARGO_AUTH_INPUT')|" $S_APP_PY

# --- Keep-Alive Setup ---
if [ -z "$HF_TOKEN" ]; then
    echo -e "\n${C_RED}WARNING: HF_TOKEN secret not found! Keep-Alive will be disabled.${C_NC}\n"
    KEEP_ALIVE_HF="false"
else
    KEEP_ALIVE_HF="true"
    HF_REPO_ID="zhengkeeling/dp"
    HF_REPO_TYPE="spaces"
fi
echo -e "${C_GREEN}Configuration applied automatically.${C_NC}"

# --- Restore Original, Working Patcher ---
echo -e "${C_BLUE}Applying original robust patch...${C_NC}"
cat > extended_patch.py << 'EOF'
# coding: utf-8
import os, base64, json, subprocess, time
with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()
old_config = 'config ={"log":{"access":"/dev/null","error":"/dev/null","loglevel":"none",},"inbounds":[{"port":ARGO_PORT ,"protocol":"vless","settings":{"clients":[{"id":UUID ,"flow":"xtls-rprx-vision",},],"decryption":"none","fallbacks":[{"dest":3001 },{"path":"/vless-argo","dest":3002 },{"path":"/vmess-argo","dest":3003 },{"path":"/trojan-argo","dest":3004 },],},"streamSettings":{"network":"tcp",},},{"port":3001 ,"listen":"127.0.0.1","protocol":"vless","settings":{"clients":[{"id":UUID },],"decryption":"none"},"streamSettings":{"network":"ws","security":"none"}},{"port":3002 ,"listen":"127.0.0.1","protocol":"vless","settings":{"clients":[{"id":UUID ,"level":0 }],"decryption":"none"},"streamSettings":{"network":"ws","security":"none","wsSettings":{"path":"/vless-argo"}},"sniffing":{"enabled":True ,"destOverride":["http","tls","quic"],"metadataOnly":False }},{"port":3003 ,"listen":"127.0.0.1","protocol":"vmess","settings":{"clients":[{"id":UUID ,"alterId":0 }]},"streamSettings":{"network":"ws","wsSettings":{"path":"/vmess-argo"}},"sniffing":{"enabled":True ,"destOverride":["http","tls","quic"],"metadataOnly":False }},{"port":3004 ,"listen":"127.0.0.1","protocol":"trojan","settings":{"clients":[{"password":UUID },]},"streamSettings":{"network":"ws","security":"none","wsSettings":{"path":"/trojan-argo"}},"sniffing":{"enabled":True ,"destOverride":["http","tls","quic"],"metadataOnly":False }},],"outbounds":[{"protocol":"freedom","tag": "direct" },{"protocol":"blackhole","tag":"block"}]}'
new_config = '''config = {
    "log": { "access": "/dev/null", "error": "/dev/null", "loglevel": "warning" },
    "inbounds": [
        { "port": ARGO_PORT, "protocol": "vless", "settings": { "clients": [{"id": UUID, "flow": "xtls-rprx-vision"}], "decryption": "none", "fallbacks": [ {"dest": 3001}, {"path": "/vless-argo", "dest": 3002}, {"path": "/vmess-argo", "dest": 3003}, {"path": "/trojan-argo", "dest": 3004} ] }, "streamSettings": {"network": "tcp"} },
        { "port": 3001, "listen": "127.0.0.1", "protocol": "vless", "settings": { "clients": [{"id": UUID}], "decryption": "none" }, "streamSettings": {"network": "ws", "security": "none"} },
        { "port": 3002, "listen": "127.0.0.1", "protocol": "vless", "settings": { "clients": [{"id": UUID, "level": 0}], "decryption": "none" }, "streamSettings": { "network": "ws", "security": "none", "wsSettings": {"path": "/vless-argo"} }, "sniffing": { "enabled": True, "destOverride": ["http", "tls", "quic"], "metadataOnly": False } },
        { "port": 3003, "listen": "127.0.0.1", "protocol": "vmess", "settings": { "clients": [{"id": UUID, "alterId": 0}] }, "streamSettings": { "network": "ws", "wsSettings": {"path": "/vmess-argo"} }, "sniffing": { "enabled": True, "destOverride": ["http", "tls", "quic"], "metadataOnly": False } },
        { "port": 3004, "listen": "127.0.0.1", "protocol": "trojan", "settings": { "clients": [{"password": UUID}] }, "streamSettings": { "network": "ws", "security": "none", "wsSettings": {"path": "/trojan-argo"} }, "sniffing": { "enabled": True, "destOverride": ["http", "tls", "quic"], "metadataOnly": False } }
    ],
    "outbounds": [
        {"protocol": "freedom", "tag": "direct"},
        { "protocol": "vmess", "tag": "media", "settings": { "vnext": [{ "address": "172.233.171.224", "port": 16416, "users": [{ "id": "8c1b9bea-cb51-43bb-a65c-0af31bbbf145", "alterId": 0 }] }] }, "streamSettings": {"network": "tcp"} },
        {"protocol": "blackhole", "tag": "block"}
    ],
    "routing": { "domainStrategy": "IPIfNonMatch", "rules": [ { "type": "field", "domain": [ "youtube.com", "youtu.be", "googlevideo.com", "ytimg.com", "gstatic.com", "googleapis.com", "ggpht.com", "googleusercontent.com", "facebook.com", "fb.com", "fbcdn.net", "instagram.com", "cdninstagram.com", "fbsbx.com", "api.facebook.com", "twitter.com", "x.com", "twimg.com", "t.co", "discord.com", "discordapp.com", "discord.gg", "discord.media", "discordapp.net", "telegram.org", "t.me", "telegram.me", "web.telegram.org", "cdn.telegram.org", "pluto.web.telegram.org", "venus.web.telegram.org", "apollo.web.telegram.org", "whatsapp.com", "whatsapp.net", "meta.com", "meta.ai", "api.meta.ai", "api.whatsapp.com", "messenger.com", "api.messenger.com", "tiktok.com", "tiktokv.com", "ttlivecdn.com", "byteoversea.com", "musical.ly", "tik-tokcdn.com", "netflix.com", "netflix.net", "nflxvideo.net", "nflximg.net", "nflxso.net", "nflxext.com" ], "outboundTag": "media" } ] }
}'''
content = content.replace(old_config, new_config)
old_generate_function = '''# Generate links and subscription content
async def generate_links(argo_domain):
    meta_info = subprocess.run(['curl', '-s', 'https://speed.cloudflare.com/meta'], capture_output=True, text=True)
    meta_info = meta_info.stdout.split('"')
    ISP = f"{meta_info[25]}-{meta_info[17]}".replace(' ', '_').strip()
    time.sleep(2)
    VMESS = {"v": "2", "ps": f"{NAME}-{ISP}", "add": CFIP, "port": CFPORT, "id": UUID, "aid": "0", "scy": "none", "net": "ws", "type": "none", "host": argo_domain, "path": "/vmess-argo?ed=2560", "tls": "tls", "sni": argo_domain, "alpn": "", "fp": "chrome"}
    list_txt = f"""
vless://{UUID}@{CFIP}:{CFPORT}?encryption=none&security=tls&sni={argo_domain}&fp=chrome&type=ws&host={argo_domain}&path=%2Fvless-argo%3Fed%3D2560#{NAME}-{ISP}
vmess://{ base64.b64encode(json.dumps(VMESS).encode('utf-8')).decode('utf-8')}
trojan://{UUID}@{CFIP}:{CFPORT}?security=tls&sni={argo_domain}&fp=chrome&type=ws&host={argo_domain}&path=%2Ftrojan-argo%3Fed%3D2560#{NAME}-{ISP}
"""
    with open(os.path.join(FILE_PATH, 'list.txt'), 'w', encoding='utf-8') as list_file: list_file.write(list_txt)
    sub_txt = base64.b64encode(list_txt.encode('utf-8')).decode('utf-8')
    with open(os.path.join(FILE_PATH, 'sub.txt'), 'w', encoding='utf-8') as sub_file: sub_file.write(sub_txt)
    print(sub_txt)
    print(f"{FILE_PATH}/sub.txt saved successfully")
    send_telegram()
    upload_nodes()
    return sub_txt'''
new_generate_function = '''# Generate links and subscription content
async def generate_links(argo_domain):
    meta_info = subprocess.run(['curl', '-s', 'https://speed.cloudflare.com/meta'], capture_output=True, text=True)
    meta_info = meta_info.stdout.split('"')
    ISP = f"{meta_info[25]}-{meta_info[17]}".replace(' ', '_').strip()
    time.sleep(1)
    VMESS_TLS = {"v": "2", "ps": f"{NAME}-{ISP}-TLS", "add": CFIP, "port": CFPORT, "id": UUID, "aid": "0", "scy": "none", "net": "ws", "type": "none", "host": argo_domain, "path": "/vmess-argo?ed=2560", "tls": "tls", "sni": argo_domain, "alpn": "", "fp": "chrome"}
    VMESS_80 = {"v": "2", "ps": f"{NAME}-{ISP}-80", "add": CFIP, "port": "80", "id": UUID, "aid": "0", "scy": "none", "net": "ws", "type": "none", "host": argo_domain, "path": "/vmess-argo?ed=2560", "tls": "", "sni": "", "alpn": "", "fp": ""}
    list_txt = f"""
vless://{UUID}@{CFIP}:{CFPORT}?encryption=none&security=tls&sni={argo_domain}&fp=chrome&type=ws&host={argo_domain}&path=%2Fvless-argo%3Fed%3D2560#{NAME}-{ISP}-TLS
vmess://{ base64.b64encode(json.dumps(VMESS_TLS).encode('utf-8')).decode('utf-8')}
trojan://{UUID}@{CFIP}:{CFPORT}?security=tls&sni={argo_domain}&fp=chrome&type=ws&host={argo_domain}&path=%2Ftrojan-argo%3Fed%3D2560#{NAME}-{ISP}-TLS
vless://{UUID}@{CFIP}:80?encryption=none&security=none&type=ws&host={argo_domain}&path=%2Fvless-argo%3Fed%3D2560#{NAME}-{ISP}-80
vmess://{ base64.b64encode(json.dumps(VMESS_80).encode('utf-8')).decode('utf-8')}
trojan://{UUID}@{CFIP}:80?security=none&type=ws&host={argo_domain}&path=%2Ftrojan-argo%3Fed%3D2560#{NAME}-{ISP}-80
"""
    with open(os.path.join(FILE_PATH, 'list.txt'), 'w', encoding='utf-8') as list_file: list_file.write(list_txt)
    sub_txt = base64.b64encode(list_txt.encode('utf-8')).decode('utf-8')
    with open(os.path.join(FILE_PATH, 'sub.txt'), 'w', encoding='utf-8') as sub_file: sub_file.write(sub_txt)
    print(sub_txt)
    print(f"{FILE_PATH}/sub.txt saved successfully")
    send_telegram()
    upload_nodes()
    return sub_txt'''
content = content.replace(old_generate_function, new_generate_function)
with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("扩展分流配置和80端口节点已成功添加")
EOF
$S_PYTHON3 extended_patch.py
rm extended_patch.py
echo -e "${C_GREEN}Patch applied successfully.${C_NC}"

# --- Start Services ---
pkill -f "$S_PYTHON3 $S_APP_PY" >/dev/null 2>&1
pkill -f "keep_alive_task.sh" >/dev/null 2>&1
sleep 2

nohup $S_PYTHON3 $S_APP_PY > app.log 2>&1 &
APP_PID=$!
sleep 2

# --- Verify Service Start ---
if ! ps -p $APP_PID > /dev/null 2>&1; then
    echo -e "${C_RED}Service failed to start. Please check the log: tail -f app.log${C_NC}"
    exit 1
fi
echo -e "${C_GREEN}Service started in background (PID: $APP_PID).${C_NC}"

# --- Start Keep-Alive if configured ---
if [ "$KEEP_ALIVE_HF" = "true" ]; then
    echo "#!/bin/bash" > keep_alive_task.sh
    echo "while true; do curl -s -o /dev/null --header \"Authorization: Bearer \$HF_TOKEN\" \"https://huggingface.co/api/$HF_REPO_TYPE/$HF_REPO_ID\"; sleep 300; done" >> keep_alive_task.sh
    chmod +x keep_alive_task.sh; nohup ./keep_alive_task.sh >/dev/null 2>&1 &
    echo -e "${C_GREEN}Keep-Alive task started.${C_NC}"
fi

# --- NEW: Intelligent Health Check Loop ---
echo -e "${C_BLUE}Waiting for node generation (max 3 minutes)...${C_NC}"
MAX_WAIT=180
ELAPSED=0
NODE_INFO=""
while [ $ELAPSED -lt $MAX_WAIT ]; do
    # Check if the python process has crashed
    if ! ps -p $APP_PID > /dev/null; then
        echo -e "\n${C_RED}ERROR: The Python service has crashed!${C_NC}"
        echo -e "${C_YELLOW}--- Last 10 lines of log (app.log): ---${C_NC}"
        tail -n 10 app.log
        exit 1
    fi

    # Check for the final success file
    if [ -f "sub.txt" ]; then
        NODE_INFO=$(cat sub.txt 2>/dev/null)
        if [ -n "$NODE_INFO" ]; then
            echo -e "\n${C_GREEN}Node information generated successfully!${C_NC}"
            break
        fi
    fi
    
    # Print a status update to the same line
    echo -n -e "\r${C_YELLOW}Elapsed: ${ELAPSED}s. Monitoring service health... ${C_NC}"

    # Wait for the next check
    sleep 10
    ELAPSED=$((ELAPSED + 10))
done

# Check if the loop timed out
if [ -z "$NODE_INFO" ]; then
    echo -e "\n${C_RED}Timeout! Node generation took too long.${C_NC}"
    echo -e "${C_YELLOW}This often means the Argo tunnel failed to connect inside the Python script.${C_NC}"
    echo -e "${C_YELLOW}--- Last 10 lines of log (app.log): ---${C_NC}"
    tail -n 10 app.log
    exit 1
fi

# --- Display Final Information ---
DECODED_NODES=$(echo "$NODE_INFO"|base64 -d 2>/dev/null||echo "Decode failed.")
echo
echo -e "${C_GREEN}========================================${C_NC}"
echo -e "${C_GREEN} Deployment Complete! ${C_NC}"
echo -e "${C_GREEN}========================================${C_NC}"
echo
echo -e "${C_YELLOW}--- Subscription Data (Base64) ---${C_NC}"
echo -e "${C_GREEN}${NODE_INFO}${C_NC}"
echo
echo -e "${C_YELLOW}--- Decoded Nodes ---${C_NC}"
echo -e "${C_GREEN}${DECODED_NODES}${C_NC}"

echo -e "--- Nodes ---\n${DECODED_NODES}\n\n--- Sub Link ---\n${NODE_INFO}" > "$NODE_INFO_FILE"
echo -e "\n${C_GREEN}All information saved to ${NODE_INFO_FILE}${C_NC}"
exit 0
