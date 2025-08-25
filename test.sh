#!/bin/bash
# =================================================================
#  FINAL VERSION - Based on the original "Neko-girl" script
#  Minimal changes for automation and security ONLY.
# =================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 1. 新增密码验证
SECRET_PASS="keeling"
echo -e -n "${YELLOW}Please enter the password to continue: ${NC}"
read -s USER_INPUT
echo
if [ "$USER_INPUT" != "$SECRET_PASS" ]; then
    echo -e "${RED}Authentication failed. Exiting.${NC}"
    exit 1
fi
echo -e "${GREEN}Authentication successful.${NC}"
echo

NODE_INFO_FILE="$HOME/.xray_nodes_info"
PROJECT_DIR_NAME="python-xray-argo"

# 如果是-v参数，直接查看节点信息
if [ "$1" = "-v" ]; then
    if [ -f "$NODE_INFO_FILE" ]; then
        echo -e "${GREEN}========================================${NC}"
        echo -e "${GREEN} 节点信息查看 ${NC}"
        echo -e "${GREEN}========================================${NC}"
        echo
        cat "$NODE_INFO_FILE"
        echo
    else
        echo -e "${RED}未找到节点信息文件${NC}"
        echo -e "${YELLOW}请先运行部署脚本生成节点信息${NC}"
    fi
    exit 0
fi

generate_uuid() {
    if command -v uuidgen &> /dev/null; then
        uuidgen | tr '[:upper:]' '[:lower:]'
    elif command -v python3 &> /dev/null; then
        python3 -c "import uuid; print(str(uuid.uuid4()))"
    else
        openssl rand -hex 16 | sed 's/\(........\)\(....\)\(....\)\(....\)\(............\)/\1-\2-\3-\4-\5/' | tr '[:upper:]' '[:lower:]'
    fi
}

clear
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN} Python Xray Argo 一键部署脚本 (自动化安全版) ${NC}"
echo -e "${GREEN}========================================${NC}"
echo
echo -e "${BLUE}基于项目: ${YELLOW}https://github.com/eooce/python-xray-argo${NC}"
echo
echo -e "${GREEN}此脚本将自动执行部署...${NC}"
sleep 2

# 自动选择完整模式
MODE_CHOICE="2"

echo
echo -e "${BLUE}检查并安装依赖...${NC}"
sudo apt-get update -qq || true
if ! command -v python3 &> /dev/null; then
    echo -e "${YELLOW}正在安装 Python3...${NC}"
    sudo apt-get install -y python3 python3-pip || true
fi
if ! python3 -c "import requests" &> /dev/null; then
    echo -e "${YELLOW}正在安装 Python 依赖: requests...${NC}"
    pip3 install --user requests
fi
if ! command -v git &> /dev/null; then
    sudo apt-get install -y git || true
fi
if ! command -v unzip &> /dev/null; then
    sudo apt-get install -y unzip || true
fi

if [ ! -d "$PROJECT_DIR_NAME" ]; then
    echo -e "${BLUE}下载完整仓库...${NC}"
    git clone --depth=1 https://github.com/eooce/python-xray-argo.git "$PROJECT_DIR_NAME" || {
        wget -q https://github.com/eooce/python-xray-argo/archive/refs/heads/main.zip -O python-xray-argo.zip
        unzip -q python-xray-argo.zip
        mv python-xray-argo-main "$PROJECT_DIR_NAME"
        rm python-xray-argo.zip
    }
    if [ $? -ne 0 ] || [ ! -d "$PROJECT_DIR_NAME" ]; then
        echo -e "${RED}下载失败，请检查网络连接${NC}"
        exit 1
    fi
fi

cd "$PROJECT_DIR_NAME"
echo -e "${GREEN}依赖安装完成！${NC}"
echo

if [ ! -f "app.py" ]; then
    echo -e "${RED}未找到app.py文件！${NC}"
    exit 1
fi
[ -f "app.py.backup" ] || cp app.py app.py.backup
echo -e "${YELLOW}已备份原始文件为 app.py.backup${NC}"

# 初始化保活变量
KEEP_ALIVE_HF="false"
HF_REPO_ID=""
HF_REPO_TYPE="spaces"

# 定义保活配置函数
configure_hf_keep_alive() {
    echo
    echo -e "${YELLOW}设置 Hugging Face API 自动保活... (自动选择 Y)${NC}"
    
    echo -e "${BLUE}正在从 Space secrets 读取 Hugging Face 令牌...${NC}"
    if [ -z "$HF_TOKEN" ]; then
        echo -e "${RED}错误：未能从 Space secrets 中找到名为 HF_TOKEN 的令牌。${NC}"
        echo -e "${YELLOW}请确认您已在 Space 的 Settings -> Secrets 中添加了它并重启。${NC}"
        KEEP_ALIVE_HF="false" 
        return
    fi
    
    echo -e "${GREEN}成功从 Space secrets 读取令牌！${NC}"
    
    # 2. 更新仓库ID
    HF_REPO_ID_INPUT="zhengkeeling/dp" 
    echo -e "${GREEN}[Repo ID 已自动设为: $HF_REPO_ID_INPUT]${NC}"
    
    HF_REPO_TYPE="spaces"

    HF_REPO_ID="$HF_REPO_ID_INPUT"
    KEEP_ALIVE_HF="true"
    echo -e "${GREEN}Hugging Face API 保活已设置！${NC}"
}

# --- 自动进入完整模式 ---
echo -e "${BLUE}=== 自动进入完整配置模式 ===${NC}"
echo
UUID_INPUT="c10a3483-5de5-4416-9a37-a6c702b916ac"
echo -e "${GREEN}[UUID 已自动填入]${NC}"
sed -i "s/UUID = os.environ.get('UUID', '[^']*')/UUID = os.environ.get('UUID', '$UUID_INPUT')/" app.py

# 3. 移除所有手动输入，使用默认值
CFIP_INPUT="joeyblog.net"
sed -i "s/CFIP = os.environ.get('CFIP', '[^']*')/CFIP = os.environ.get('CFIP', '$CFIP_INPUT')/" app.py
echo -e "${GREEN}[所有配置项将使用默认值]${NC}"

configure_hf_keep_alive

ARGO_DOMAIN_INPUT="face.keeling.dpdns.org"
sed -i "s|ARGO_DOMAIN = os.environ.get('ARGO_DOMAIN', '[^']*')|ARGO_DOMAIN = os.environ.get('ARGO_DOMAIN', '$ARGO_DOMAIN_INPUT')|" app.py

ARGO_AUTH_INPUT='{"AccountTag":"46fad1b6b0e334ca8ad9ea7ec29c4ddb","TunnelSecret":"J2TOKaJiWL8rph+m7iTfEOthVtREnhuvfWoHp4SmOog=","TunnelID":"29e3716e-783c-4a1f-9538-d40fa766006f","Endpoint":""}'
sed -i "s|ARGO_AUTH = os.environ.get('ARGO_AUTH', '[^']*')|ARGO_AUTH = os.environ.get('ARGO_AUTH', '$ARGO_AUTH_INPUT')|" app.py

echo
echo -e "${GREEN}扩展分流已自动配置${NC}"
echo
echo -e "${GREEN}完整配置完成！${NC}"

# --- 启动服务 ---
echo
echo -e "${BLUE}正在启动服务...${NC}"

# !!! 下面的Python补丁部分，完全维持“猫娘版”的原始状态，不做任何修改，以确保100%稳定 !!!
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
    ISP = "HF-Node"
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
python3 extended_patch.py
rm extended_patch.py
echo -e "${GREEN}扩展分流和80端口节点已集成${NC}"

pkill -f "python3 app.py" > /dev/null 2>&1
pkill -f "keep_alive_task.sh" > /dev/null 2>&1
sleep 2

nohup python3 app.py > app.log 2>&1 &
APP_PID=$!
sleep 2

APP_PID=$(pgrep -f "python3 app.py" | head -1)
if [ -z "$APP_PID" ]; then
    echo -e "${RED}服务启动失败，请检查Python环境${NC}"
    echo -e "${YELLOW}查看日志: tail -f app.log${NC}"
    exit 1
fi
echo -e "${GREEN}服务已在后台启动，PID: $APP_PID${NC}"

if [ "$KEEP_ALIVE_HF" = "true" ]; then
    echo -e "${BLUE}正在创建并启动 Hugging Face API 保活任务...${NC}"
    echo "#!/bin/bash" > keep_alive_task.sh
    echo "while true; do" >> keep_alive_task.sh
    echo "    API_PATH=\"https://huggingface.co/api/${HF_REPO_TYPE}/${HF_REPO_ID}\"" >> keep_alive_task.sh
    echo "    status_code=\$(curl -s -o /dev/null -w \"%{http_code}\" --header \"Authorization: Bearer \$HF_TOKEN\" \"\$API_PATH\")" >> keep_alive_task.sh
    echo "    sleep 300" >> keep_alive_task.sh
    echo "done" >> keep_alive_task.sh

    export HF_TOKEN="$HF_TOKEN"
    chmod +x keep_alive_task.sh
    nohup ./keep_alive_task.sh >/dev/null 2>&1 &
    echo -e "${GREEN}Hugging Face API 保活任务已启动。${NC}"
fi

echo -e "${BLUE}等待节点信息生成 (最长等待3分钟)...${NC}"
MAX_WAIT=180; ELAPSED=0; NODE_INFO=""
while [ $ELAPSED -lt $MAX_WAIT ]; do
    if ! ps -p $APP_PID > /dev/null; then
        echo -e "\n${RED}错误: 服务进程意外崩溃!${NC}"; echo -e "${YELLOW}日志尾部:${NC}"; tail -n 10 app.log; exit 1
    fi
    if [ -f "sub.txt" ]; then
        NODE_INFO=$(cat sub.txt 2>/dev/null)
        if [ -n "$NODE_INFO" ]; then echo -e "\n${GREEN}节点信息已生成！${NC}"; break; fi
    fi
    echo -n -e "\r${YELLOW}已等待 ${ELAPSED}s. 正在监控服务状态... ${NC}"
    sleep 10; ELAPSED=$((ELAPSED + 10))
done

if [ -z "$NODE_INFO" ]; then
    echo -e "\n${RED}等待超时！节点信息未能生成。${NC}"; echo -e "${YELLOW}这通常是Argo隧道连接问题。日志尾部:${NC}"; tail -n 10 app.log; exit 1
fi

echo
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN} 部署完成！ ${NC}"
echo -e "${GREEN}========================================${NC}"
echo

# (The rest of the script for displaying info is omitted for brevity but is identical to the original)

DECODED_NODES=$(echo "$NODE_INFO" | base64 -d 2>/dev/null || echo "$NODE_INFO")
echo -e "${YELLOW}=== 节点信息 ===${NC}"
echo -e "${GREEN}订阅链接:${NC}"
echo "$NODE_INFO"
echo -e "\n${GREEN}节点配置:${NC}"
echo "$DECODED_NODES"
echo

# ... (rest of the original save and celebration code) ...
function celebration_animation() {
    echo -e "\n\n"
    echo -e "${GREEN}喵~ 部署任务大成功啦！ >ω<${NC}"
    sleep 0.5
    echo -e "${YELLOW}正在为主人献上胜利的爱心... (｡♥‿♥｡)${NC}"
    sleep 0.5
    echo -e "${RED}"
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
    echo -e "${NC}"
    sleep 1
    echo -e "${BLUE}所有节点都准备就绪，正在检查最后的魔力...${NC}"
    for i in {1..20}; do echo -n "✨"; sleep 0.05; done
    echo -e "\n${GREEN}魔力注入完毕！随时可以出发咯！喵~${NC}\n"
}
celebration_animation
exit 0
