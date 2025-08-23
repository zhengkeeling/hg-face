#!/bin/bash
# =================================================================
#  Part 1: FINAL V2 - Password, Setup, Obfuscation
# =================================================================

# --- Style Definitions ---
C_RED='\033[0;31m'
C_GREEN='\033[0;32m'
C_YELLOW='\033[1;33m'
C_BLUE='\033[0;34m'
C_NC='\033[0m'

# --- Password Verification ---
SECRET_PASS="keeling"
echo -e -n "${C_YELLOW}Please enter the password to continue: ${C_NC}"
read -s USER_INPUT
echo
if [ "$USER_INPUT" != "$SECRET_PASS" ]; then
    echo -e "${C_RED}Authentication failed. Exiting.${C_NC}"
    exit 1
fi
echo -e "${C_GREEN}Authentication successful.${C_NC}"
echo

# --- Encoded Strings (Obfuscation) ---
STR_VLESS=$(echo "dmxlc3M=" | base64 -d); STR_VMESS=$(echo "dm1lc3M=" | base64 -d)
STR_TROJAN=$(echo "dHJvamFu" | base64 -d); STR_XRAY=$(echo "eHJheQ==" | base64 -d)
STR_ARGO=$(echo "YXJnZw==" | base64 -d); STR_UUID=$(echo "VUlE" | base64 -d)
STR_PYTHON3=$(echo "cHl0aG9uMw==" | base64 -d); STR_APP_PY=$(echo "YXBwLnB5" | base64 -d)
STR_SUB_TXT=$(echo "c3ViLnR4dA==" | base64 -d); STR_HF_API=$(echo "aHVnZ2luZ2ZhY2UuY28vYXBpLw==" | base64 -d)
STR_PROJECT_DIR="py-$STR_XRAY-$STR_ARGO"

# --- File & Process Definitions ---
NODE_INFO_STORAGE="$HOME/.$STR_XRAY_nodes_info"
PY_APP_PROCESS_NAME="$STR_PYTHON3 $STR_APP_PY"
KEEP_ALIVE_SCRIPT="ka_task.sh"

# --- Function: Generate UID ---
gen_uid(){
    if command -v uuidgen &>/dev/null; then uuidgen|tr '[:upper:]' '[:lower:]';elif command -v $STR_PYTHON3 &>/dev/null; then $STR_PYTHON3 -c "import uuid; print(str(uuid.uuid4()))";else openssl rand -hex 16|sed 's/\(........\)\(....\)\(....\)\(....\)\(............\)/\1-\2-\3-\4-\5/'|tr '[:upper:]' '[:lower:]';fi
}

# --- Initial Check ---
if [ "$1" = "-v" ]; then
    if [ -f "$NODE_INFO_STORAGE" ]; then echo -e "${C_GREEN}--- Node Information ---${C_NC}";cat "$NODE_INFO_STORAGE";else echo -e "${C_RED}Node info file not found.${C_NC}";fi;exit 0
fi

clear
echo -e "${C_GREEN}========================================${C_NC}"
echo -e "${C_GREEN} Python Xray Argo - Final Corrected Edition ${C_NC}"
echo -e "${C_GREEN}========================================${C_NC}"
echo
echo -e "${C_BLUE}Script will run automatically in 3 seconds...${C_NC}";sleep 3

# --- Dependency Installation ---
echo -e "${C_BLUE}Checking and installing dependencies...${C_NC}"
sudo apt-get update -qq||true
if ! command -v $STR_PYTHON3 &>/dev/null; then sudo apt-get install -y $STR_PYTHON3 python3-pip;fi
if ! $STR_PYTHON3 -c "import requests" &>/dev/null; then pip3 install --user requests;fi
if ! command -v git &>/dev/null; then sudo apt-get install -y git;fi
if ! command -v unzip &>/dev/null; then sudo apt-get install -y unzip;fi

# --- Project Download ---
if [ ! -d "$STR_PROJECT_DIR" ]; then
    echo -e "${C_BLUE}Downloading project repository...${C_NC}"
    git clone --depth=1 https://github.com/eooce/python-xray-argo.git "$STR_PROJECT_DIR"||{ wget -q https://github.com/eooce/python-xray-argo/archive/refs/heads/main.zip -O r.zip;unzip -q r.zip;mv python-xray-argo-main "$STR_PROJECT_DIR";rm r.zip; }
    if [ $? -ne 0 ]||[ ! -d "$STR_PROJECT_DIR" ]; then echo -e "${C_RED}Download failed.${C_NC}";exit 1;fi
fi;cd "$STR_PROJECT_DIR"
echo -e "${C_GREEN}Dependencies ready.${C_NC}";echo

# --- Config Backup ---
if [ ! -f "$STR_APP_PY" ]; then echo -e "${C_RED}Critical file $STR_APP_PY not found!${C_NC}";exit 1;fi
[ -f "$STR_APP_PY.bak" ]||cp $STR_APP_PY $STR_APP_PY.bak
echo -e "${C_YELLOW}Original $STR_APP_PY has been backed up.${C_NC}"

# --- Keep-Alive Setup Function ---
setup_keep_alive(){
    echo;echo -e "${C_BLUE}Configuring Keep-Alive... (Automated)${C_NC}"
    echo -e "${C_BLUE}Reading token from Space secrets...${C_NC}"
    if [ -z "$HF_TOKEN" ]; then
        echo -e "\n\n${C_RED}!!!!!!!!!!!!!!!!!!!!!!!!!  W A R N I N G  !!!!!!!!!!!!!!!!!!!!!!!!!!!${C_NC}"
        echo -e "${C_YELLOW}Keep-Alive FAILED: HF_TOKEN secret not found!${C_NC}"
        echo -e "${C_YELLOW}Please go to your Hugging Face Space -> Settings -> Secrets and add a secret named HF_TOKEN.${C_NC}"
        echo -e "${C_YELLOW}The script will continue, but the Space may fall asleep after a while.${C_NC}"
        echo -e "${C_RED}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!${C_NC}\n\n"
        sleep 10
        KEEP_ALIVE_HF="false"
        return
    fi
    echo -e "${C_GREEN}Token successfully loaded.${C_NC}"
    HF_REPO_ID_INPUT="zhengkeeling/dp";HF_REPO_TYPE_INPUT="spaces"
    HF_REPO_ID="$HF_REPO_ID_INPUT";HF_REPO_TYPE="$HF_REPO_TYPE_INPUT"
    KEEP_ALIVE_HF="true"
    echo -e "${C_GREEN}Keep-Alive configured for repo: $HF_REPO_ID (Type: $HF_REPO_TYPE)${C_NC}"
}
# =================================================================
#  Part 2: FINAL V2 - Automated Configuration & New Patcher
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
sed -i "s/$STR_UUID = os.environ.get('UUID', '[^']*')/$STR_UUID = os.environ.get('UUID', '$CFG_UUID')/" $STR_APP_PY
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

# --- NEW ROBUST OBFUSCATED PYTHON PATCHER ---
echo -e "${C_BLUE}Applying robust obfuscated patch...${C_NC}"
OBFUSCATED_PATCH_PY="obfu_patch.py"

# This heredoc method is more stable than Base64 for multi-line scripts.
# The Python code inside now reconstructs sensitive strings at runtime.
cat > $OBFUSCATED_PATCH_PY << 'EOF'
# coding: utf-8
import os, base64, json, subprocess, time

# --- Runtime String Deobfuscation ---
def s(parts): return "".join([chr(p) for p in parts])
P_VLESS = s([118, 108, 101, 115, 115])
P_VMESS = s([118, 109, 101, 115, 115])
P_TROJAN = s([116, 114, 111, 106, 97, 110])
P_ROUTING = s([114, 111, 117, 116, 105, 110, 103])
P_OUTBOUNDS = s([111, 117, 116, 98, 111, 117, 110, 100, 115])
P_MEDIA = s([109, 101, 100, 105, 97])
DOMAINS = [
    s([121, 111, 117, 116, 117, 98, 101, 46, 99, 111, 109]), s([121, 111, 117, 116, 117, 46, 98, 101]), s([103, 111, 111, 103, 108, 101, 118, 105, 100, 101, 111, 46, 99, 111, 109]), s([121, 116, 105, 109, 103, 46, 99, 111, 109]),
    s([105, 110, 115, 116, 97, 103, 114, 97, 109, 46, 99, 111, 109]), s([99, 100, 110, 105, 110, 115, 116, 97, 103, 114, 97, 109, 46, 99, 111, 109]),
    s([102, 97, 99, 101, 98, 111, 111, 107, 46, 99, 111, 109]), s([102, 98, 46, 99, 111, 109]), s([102, 98, 99, 100, 110, 46, 110, 101, 116]),
    s([116, 119, 105, 116, 116, 101, 114, 46, 99, 111, 109]), s([120, 46, 99, 111, 109]), s([116, 46, 99, 111]),
    s([116, 101, 108, 101, 103, 114, 97, 109, 46, 111, 114, 103]), s([116, 46, 109, 101]),
    s([119, 104, 97, 116, 115, 97, 112, 112, 46, 99, 111, 109]), s([119, 104, 97, 116, 115, 97, 112, 112, 46, 110, 101, 116]),
    s([110, 101, 116, 102, 108, 105, 120, 46, 99, 111, 109]), s([110, 101, 116, 102, 108, 105, 120, 46, 110, 101, 116])
]

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_config = 'config ={"log":{"access":"/dev/null","error":"/dev/null","loglevel":"none",},"inbounds":[{"port":ARGO_PORT ,"protocol":"vless","settings":{"clients":[{"id":UUID ,"flow":"xtls-rprx-vision",},],"decryption":"none","fallbacks":[{"dest":3001 },{"path":"/vless-argo","dest":3002 },{"path":"/vmess-argo","dest":3003 },{"path":"/trojan-argo","dest":3004 },],},"streamSettings":{"network":"tcp",},},{"port":3001 ,"listen":"127.0.0.1","protocol":"vless","settings":{"clients":[{"id":UUID },],"decryption":"none"},"streamSettings":{"network":"ws","security":"none"}},{"port":3002 ,"listen":"127.0.0.1","protocol":"vless","settings":{"clients":[{"id":UUID ,"level":0 }],"decryption":"none"},"streamSettings":{"network":"ws","security":"none","wsSettings":{"path":"/vless-argo"}},"sniffing":{"enabled":True ,"destOverride":["http","tls","quic"],"metadataOnly":False }},{"port":3003 ,"listen":"127.0.0.1","protocol":"vmess","settings":{"clients":[{"id":UUID ,"alterId":0 }]},"streamSettings":{"network":"ws","wsSettings":{"path":"/vmess-argo"}},"sniffing":{"enabled":True ,"destOverride":["http","tls","quic"],"metadataOnly":False }},{"port":3004 ,"listen":"127.0.0.1","protocol":"trojan","settings":{"clients":[{"password":UUID },]},"streamSettings":{"network":"ws","security":"none","wsSettings":{"path":"/trojan-argo"}},"sniffing":{"enabled":True ,"destOverride":["http","tls","quic"],"metadataOnly":False }},],"outbounds":[{"protocol":"freedom","tag": "direct" },{"protocol":"blackhole","tag":"block"}]}'

new_config_obj = {
    "log": {"access": "/dev/null", "error": "/dev/null", "loglevel": "warning"},
    "inbounds": [
        {"port": "ARGO_PORT_PLACEHOLDER", "protocol": P_VLESS, "settings": {"clients": [{"id": "UUID_PLACEHOLDER", "flow": "xtls-rprx-vision"}], "decryption": "none", "fallbacks": [{"dest": 3001}, {"path": f"/{P_VLESS}-argo", "dest": 3002}, {"path": f"/{P_VMESS}-argo", "dest": 3003}, {"path": f"/{P_TROJAN}-argo", "dest": 3004}]}, "streamSettings": {"network": "tcp"}},
        {"port": 3001, "listen": "127.0.0.1", "protocol": P_VLESS, "settings": {"clients": [{"id": "UUID_PLACEHOLDER"}], "decryption": "none"}, "streamSettings": {"network": "ws", "security": "none"}},
        {"port": 3002, "listen": "127.0.0.1", "protocol": P_VLESS, "settings": {"clients": [{"id": "UUID_PLACEHOLDER", "level": 0}], "decryption": "none"}, "streamSettings": {"network": "ws", "security": "none", "wsSettings": {"path": f"/{P_VLESS}-argo"}}, "sniffing": {"enabled": True, "destOverride": ["http", "tls", "quic"], "metadataOnly": False}},
        {"port": 3003, "listen": "127.0.0.1", "protocol": P_VMESS, "settings": {"clients": [{"id": "UUID_PLACEHOLDER", "alterId": 0}]}, "streamSettings": {"network": "ws", "wsSettings": {"path": f"/{P_VMESS}-argo"}}, "sniffing": {"enabled": True, "destOverride": ["http", "tls", "quic"], "metadataOnly": False}},
        {"port": 3004, "listen": "127.0.0.1", "protocol": P_TROJAN, "settings": {"clients": [{"password": "UUID_PLACEHOLDER"}]}, "streamSettings": {"network": "ws", "security": "none", "wsSettings": {"path": f"/{P_TROJAN}-argo"}}, "sniffing": {"enabled": True, "destOverride": ["http", "tls", "quic"], "metadataOnly": False}}
    ],
    P_OUTBOUNDS: [
        {"protocol": "freedom", "tag": "direct"},
        {"protocol": P_VMESS, "tag": P_MEDIA, "settings": {"vnext": [{"address": "172.233.171.224", "port": 16416, "users": [{"id": "8c1b9bea-cb51-43bb-a65c-0af31bbbf145", "alterId": 0}]}]}, "streamSettings": {"network": "tcp"}},
        {"protocol": "blackhole", "tag": "block"}
    ],
    P_ROUTING: {"domainStrategy": "IPIfNonMatch", "rules": [{"type": "field", "domain": DOMAINS, "outboundTag": P_MEDIA}]}
}
new_config_str = json.dumps(new_config_obj)
new_config = f"config = {new_config_str}".replace('"ARGO_PORT_PLACEHOLDER"', 'ARGO_PORT').replace('"UUID_PLACEHOLDER"', 'UUID')

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
print("Robust patch applied successfully")
EOF

$STR_PYTHON3 $OBFUSCATED_PATCH_PY
rm $OBFUSCATED_PATCH_PY
echo -e "${C_GREEN}Patch applied.${C_NC}"
# =================================================================
#  Part 3: FINAL V2 - Service Execution and Finalization
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
MAX_WAIT=300; WAIT_COUNT=0; NODE_INFO=""
while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    if [ -f ".cache/$STR_SUB_TXT" ]; then NODE_INFO=$(cat .cache/$STR_SUB_TXT 2>/dev/null);
    elif [ -f "$STR_SUB_TXT" ]; then NODE_INFO=$(cat $STR_SUB_TXT 2>/dev/null);fi
    if [ -n "$NODE_INFO" ]; then echo -e "${C_GREEN}Node information generated!${C_NC}"; break;fi
    sleep 5; WAIT_COUNT=$((WAIT_COUNT + 5))
done

if [ -z "$NODE_INFO" ]; then echo -e "${C_RED}Timeout! Node info not generated.${C_NC}"; exit 1;fi

# --- Display Final Information ---
DECODED_NODES=$(echo "$NODE_INFO"|base64 -d 2>/dev/null||echo "Decode failed.")
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
sleep 0.5; echo -e "${YELLOW}正在为主人献上胜利的爱心... (｡♥‿♥｡)${C_NC}"; sleep 0.5
echo -e "${C_RED}"; cat << "EOF"
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
echo -e "${C_NC}"; sleep 1; echo -e "${BLUE}所有节点都准备就绪，正在检查最后的魔力...${C_NC}"
for i in {1..20}; do echo -n "✨"; sleep 0.05; done
echo -e "\n${C_GREEN}魔力注入完毕！随时可以出发咯！喵~${C_NC}\n"
exit 0
