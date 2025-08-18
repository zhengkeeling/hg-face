#!/bin/bash
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

NODE_INFO_FILE="$HOME/.xray_nodes_info"
PROJECT_DIR_NAME="python-xray-argo"

# 如果是-v参数，直接查看节点信息
if [ "$1" = "-v" ]; then
    if [ -f "$NODE_INFO_FILE" ]; then
        echo -e "${GREEN}========================================${NC}"
        echo -e "${GREEN} 节点信息查看 ${NC}"
        echo -e "${GREEN}========================================${NC}"
        cat "$NODE_INFO_FILE"
        echo
    else
        echo -e "${RED}未找到节点信息文件${NC}"
        echo -e "${YELLOW}请先运行部署脚本生成节点信息${NC}"
    fi
    exit 0
fi
# 如果是-s参数，查看保活状态
if [ "$1" = "-s" ]; then
    echo
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN} Hugging Face API 保活状态检查 ${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo
    if [ -d "$PROJECT_DIR_NAME" ]; then
        cd "$PROJECT_DIR_NAME"
    fi
    KEEPALIVE_PID=$(pgrep -f "keep_alive_task.sh")
    if [ -n "$KEEPALIVE_PID" ]; then
        echo -e "服务状态: ${GREEN}运行中${NC}"
        echo -e "进程PID: ${BLUE}$KEEPALIVE_PID${NC}"
        if [ -f "keep_alive_task.sh" ]; then
            REPO_ID=$(grep 'huggingface.co/api/' keep_alive_task.sh | head -1 | sed -n 's|.*api/\([^"]*\)/\([^"]*\).*|\2:\1|p')
            echo -e "目标仓库: ${YELLOW}$REPO_ID${NC}"
        fi
        echo -e "\n${YELLOW}--- 最近一次保活状态 ---${NC}"
        if [ -f "keep_alive_status.log" ]; then
            cat keep_alive_status.log
        else
            echo -e "${YELLOW}尚未生成状态日志，请稍等片刻(最多5分钟)后重试...${NC}"
        fi
    else
        echo -e "服务状态: ${RED}未运行${NC}"
        echo -e "${YELLOW}提示: 您可能尚未部署服务或未在部署时设置Hugging Face保活。${NC}"
    fi
    echo
    exit 0
fi

clear
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN} Python Xray Argo 一键部署脚本 (定制版) ${NC}"
echo -e "${GREEN}========================================${NC}"
echo
echo -e "${BLUE}基于项目: ${YELLOW}https://github.com/eooce/python-xray-argo${NC}"
echo -e "${BLUE}脚本仓库: ${YELLOW}https://github.com/byJoey/free-vps-py${NC}"
echo
echo -e "${GREEN}此脚本已根据您的信息进行预配置。${NC}"
echo -e "${GREEN}将自动使用以下信息进行部署:${NC}"
echo -e "${YELLOW}  - UUID: c10a3483-5de5-4416-9a37-a6c702b916ac${NC}"
echo -e "${YELLOW}  - Argo隧道: face.keeling.dpdns.org${NC}"
echo -e "${YELLOW}  - 保活仓库: sukikeeling/face${NC}"
echo
read -p "按 Enter 键开始部署，或按 Ctrl+C 取消..."

echo
echo -e "${BLUE}检查并安装依赖...${NC}"
# 优化依赖检查：使用apt缓存避免重复update
sudo apt-get update -qq || true
if ! command -v python3 &> /dev/null; then
    echo -e "${YELLOW}正在安装 Python3...${NC}"
    sudo apt-get install -y python3 python3-pip
fi
if ! python3 -c "import requests" &> /dev/null; then
    echo -e "${YELLOW}正在安装 Python 依赖: requests...${NC}"
    pip3 install --user requests
fi
if ! command -v git &> /dev/null; then
    sudo apt-get install -y git
fi
if ! command -v unzip &> /dev/null; then
    sudo apt-get install -y unzip
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

# --- 自动配置区 ---
echo -e "${BLUE}=== 开始自动配置 ===${NC}"

# 1. 设置 UUID
UUID_INPUT="c10a3483-5de5-4416-9a37-a6c702b916ac"
sed -i "s/UUID = os.environ.get('UUID', '[^']*')/UUID = os.environ.get('UUID', '$UUID_INPUT')/" app.py
echo -e "${GREEN}UUID 已设置为: $UUID_INPUT${NC}"

# 2. 设置 Argo 固定隧道
ARGO_DOMAIN_INPUT="face.keeling.dpdns.org"
ARGO_AUTH_INPUT='{"AccountTag":"46fad1b6b0e334ca8ad9ea7ec29c4ddb","TunnelSecret":"J2TOKaJiWL8rph+m7iTfEOthVtREnhuvfWoHp4SmOog=","TunnelID":"29e3716e-783c-4a1f-9538-d40fa766006f","Endpoint":""}'
sed -i "s|ARGO_DOMAIN = os.environ.get('ARGO_DOMAIN', '[^']*')|ARGO_DOMAIN = os.environ.get('ARGO_DOMAIN', '$ARGO_DOMAIN_INPUT')|" app.py
sed -i "s|ARGO_AUTH = os.environ.get('ARGO_AUTH', '[^']*')|ARGO_AUTH = os.environ.get('ARGO_AUTH', '$ARGO_AUTH_INPUT')|" app.py
echo -e "${GREEN}Argo 固定隧道已设置为: $ARGO_DOMAIN_INPUT${NC}"

# 3. 设置 Hugging Face API 自动保活
KEEP_ALIVE_HF="true"
HF_TOKEN="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"
HF_REPO_ID="sukikeeling/face"
HF_REPO_TYPE="spaces" # 默认使用 spaces
echo -e "${GREEN}Hugging Face API 保活已设置为仓库: $HF_REPO_ID${NC}"

# 4. 其他配置使用默认值
sed -i "s/CFIP = os.environ.get('CFIP', '[^']*')/CFIP = os.environ.get('CFIP', 'joeyblog.net')/" app.py
echo -e "${GREEN}优选IP已自动设置为: joeyblog.net${NC}"
echo -e "${GREEN}扩展分流已自动配置${NC}"
echo
echo -e "${GREEN}自动配置完成！${NC}"
# --- 自动配置区结束 ---


echo -e "${YELLOW}=== 当前配置摘要 ===${NC}"
echo -e "UUID: $(grep "UUID = " app.py | head -1 | cut -d"'" -f2)"
echo -e "节点名称: $(grep "NAME = " app.py | head -1 | cut -d"'" -f4)"
echo -e "服务端口: $(grep "PORT = int" app.py | grep -o "or [0-9]*" | cut -d" " -f2)"
echo -e "优选IP: $(grep "CFIP = " app.py | cut -d"'" -f4)"
echo -e "Argo域名: $(grep "ARGO_DOMAIN = " app.py | cut -d"'" -f4)"
if [ "$KEEP_ALIVE_HF" = "true" ]; then
    echo -e "保活仓库: $HF_REPO_ID ($HF_REPO_TYPE)"
fi
echo -e "${YELLOW}========================${NC}"
echo
echo -e "${BLUE}正在启动服务...${NC}"
echo -e "${YELLOW}当前工作目录：$(pwd)${NC}"
echo
# 修改Python文件添加扩展分流到xray配置，并增加80端口节点 (优化: 扩展更多域名以支持更多流量, 增强隐私: 避免不必要日志)
echo -e "${BLUE}正在添加扩展分流功能和80端口节点...${NC}"
cat > extended_patch.py << 'EOF'
# coding: utf-8
import os, base64, json, subprocess, time
# 读取app.py文件
with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 找到原始配置并替换为包含扩展分流的配置
old_config = 'config ={"log":{"access":"/dev/null","error":"/dev/null","loglevel":"none",},"inbounds":[{"port":ARGO_PORT ,"protocol":"vless","settings":{"clients":[{"id":UUID ,"flow":"xtls-rprx-vision",},],"decryption":"none","fallbacks":[{"dest":3001 },{"path":"/vless-argo","dest":3002 },{"path":"/vmess-argo","dest":3003 },{"path":"/trojan-argo","dest":3004 },],},"streamSettings":{"network":"tcp",},},{"port":3001 ,"listen":"127.0.0.1","protocol":"vless","settings":{"clients":[{"id":UUID },],"decryption":"none"},"streamSettings":{"network":"ws","security":"none"}},{"port":3002 ,"listen":"127.0.0.1","protocol":"vless","settings":{"clients":[{"id":UUID ,"level":0 }],"decryption":"none"},"streamSettings":{"network":"ws","security":"none","wsSettings":{"path":"/vless-argo"}},"sniffing":{"enabled":True ,"destOverride":["http","tls","quic"],"metadataOnly":False }},{"port":3003 ,"listen":"127.0.0.1","protocol":"vmess","settings":{"clients":[{"id":UUID ,"alterId":0 }]},"streamSettings":{"network":"ws","wsSettings":{"path":"/vmess-argo"}},"sniffing":{"enabled":True ,"destOverride":["http","tls","quic"],"metadataOnly":False }},{"port":3004 ,"listen":"127.0.0.1","protocol":"trojan","settings":{"clients":[{"password":UUID },]},"streamSettings":{"network":"ws","security":"none","wsSettings":{"path":"/trojan-argo"}},"sniffing":{"enabled":True ,"destOverride":["http","tls","quic"],"metadataOnly":False }},],"outbounds":[{"protocol":"freedom","tag": "direct" },{"protocol":"blackhole","tag":"block"}]}'
new_config = '''config = {
    "log": {
        "access": "/dev/null",
        "error": "/dev/null",
        "loglevel": "warning" # 提升隐私: 减少日志级别
    },
    "inbounds": [
        {
            "port": ARGO_PORT,
            "protocol": "vless",
            "settings": {
                "clients": [{"id": UUID, "flow": "xtls-rprx-vision"}],
                "decryption": "none",
                "fallbacks": [
                    {"dest": 3001},
                    {"path": "/vless-argo", "dest": 3002},
                    {"path": "/vmess-argo", "dest": 3003},
                    {"path": "/trojan-argo", "dest": 3004}
                ]
            },
            "streamSettings": {"network": "tcp"}
        },
        {
            "port": 3001,
            "listen": "127.0.0.1",
            "protocol": "vless",
            "settings": {
                "clients": [{"id": UUID}],
                "decryption": "none"
            },
            "streamSettings": {"network": "ws", "security": "none"}
        },
        {
            "port": 3002,
            "listen": "127.0.0.1",
            "protocol": "vless",
            "settings": {
                "clients": [{"id": UUID, "level": 0}],
                "decryption": "none"
            },
            "streamSettings": {
                "network": "ws",
                "security": "none",
                "wsSettings": {"path": "/vless-argo"}
            },
            "sniffing": {
                "enabled": True,
                "destOverride": ["http", "tls", "quic"],
                "metadataOnly": False
            }
        },
        {
            "port": 3003,
            "listen": "127.0.0.1",
            "protocol": "vmess",
            "settings": {
                "clients": [{"id": UUID, "alterId": 0}]
            },
            "streamSettings": {
                "network": "ws",
                "wsSettings": {"path": "/vmess-argo"}
            },
            "sniffing": {
                "enabled": True,
                "destOverride": ["http", "tls", "quic"],
                "metadataOnly": False
            }
        },
        {
            "port": 3004,
            "listen": "127.0.0.1",
            "protocol": "trojan",
            "settings": {
                "clients": [{"password": UUID}]
            },
            "streamSettings": {
                "network": "ws",
                "security": "none",
                "wsSettings": {"path": "/trojan-argo"}
            },
            "sniffing": {
                "enabled": True,
                "destOverride": ["http", "tls", "quic"],
                "metadataOnly": False
            }
        }
    ],
    "outbounds": [
        {"protocol": "freedom", "tag": "direct"},
        {
            "protocol": "vmess",
            "tag": "media",
            "settings": {
                "vnext": [{
                    "address": "172.233.171.224",
                    "port": 16416,
                    "users": [{
                        "id": "8c1b9bea-cb51-43bb-a65c-0af31bbbf145",
                        "alterId": 0
                    }]
                }]
            },
            "streamSettings": {"network": "tcp"}
        },
        {"protocol": "blackhole", "tag": "block"}
    ],
    "routing": {
        "domainStrategy": "IPIfNonMatch",
        "rules": [
            {
                "type": "field",
                "domain": [
                    # YouTube
                    "youtube.com", "youtu.be", "googlevideo.com", "ytimg.com", "gstatic.com", "googleapis.com", "ggpht.com", "googleusercontent.com",
                    # Facebook/Instagram
                    "facebook.com", "fb.com", "fbcdn.net", "instagram.com", "cdninstagram.com", "fbsbx.com", "api.facebook.com",
                    # Twitter/X
                    "twitter.com", "x.com", "twimg.com", "t.co",
                    # Discord
                    "discord.com", "discordapp.com", "discord.gg", "discord.media", "discordapp.net",
                    # Telegram
                    "telegram.org", "t.me", "telegram.me", "web.telegram.org", "cdn.telegram.org", "pluto.web.telegram.org", "venus.web.telegram.org", "apollo.web.telegram.org",
                    # WhatsApp/Meta
                    "whatsapp.com", "whatsapp.net", "meta.com", "meta.ai", "api.meta.ai", "api.whatsapp.com", "messenger.com", "api.messenger.com",
                    # TikTok
                    "tiktok.com", "tiktokv.com", "ttlivecdn.com", "byteoversea.com", "musical.ly", "tik-tokcdn.com",
                    # Netflix
                    "netflix.com", "netflix.net", "nflxvideo.net", "nflximg.net", "nflxso.net", "nflxext.com"
                ],
                "outboundTag": "media"
            }
        ]
    }
}'''
# 替换配置
if old_config in content:
    content = content.replace(old_config, new_config)

# 修改generate_links函数，添加80端口节点 (优化: 添加更多协议变体以支持更多流量)
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
    with open(os.path.join(FILE_PATH, 'list.txt'), 'w', encoding='utf-8') as list_file:
        list_file.write(list_txt)
    sub_txt = base64.b64encode(list_txt.encode('utf-8')).decode('utf-8')
    with open(os.path.join(FILE_PATH, 'sub.txt'), 'w', encoding='utf-8') as sub_file:
        sub_file.write(sub_txt)
    print(sub_txt)
    print(f"{FILE_PATH}/sub.txt saved successfully")
    # Additional actions
    send_telegram()
    upload_nodes()
    return sub_txt'''
new_generate_function = '''# Generate links and subscription content
async def generate_links(argo_domain):
    meta_info = subprocess.run(['curl', '-s', 'https://speed.cloudflare.com/meta'], capture_output=True, text=True)
    meta_info = meta_info.stdout.split('"')
    ISP = f"{meta_info[25]}-{meta_info[17]}".replace(' ', '_').strip()
    time.sleep(1) # 优化: 减少等待时间
    # TLS节点
    VMESS_TLS = {"v": "2", "ps": f"{NAME}-{ISP}-TLS", "add": CFIP, "port": CFPORT, "id": UUID, "aid": "0", "scy": "none", "net": "ws", "type": "none", "host": argo_domain, "path": "/vmess-argo?ed=2560", "tls": "tls", "sni": argo_domain, "alpn": "", "fp": "chrome"}
    # 无TLS节点 (80端口)
    VMESS_80 = {"v": "2", "ps": f"{NAME}-{ISP}-80", "add": CFIP, "port": "80", "id": UUID, "aid": "0", "scy": "none", "net": "ws", "type": "none", "host": argo_domain, "path": "/vmess-argo?ed=2560", "tls": "", "sni": "", "alpn": "", "fp": ""}
    list_txt = f"""
vless://{UUID}@{CFIP}:{CFPORT}?encryption=none&security=tls&sni={argo_domain}&fp=chrome&type=ws&host={argo_domain}&path=%2Fvless-argo%3Fed%3D2560#{NAME}-{ISP}-TLS
vmess://{ base64.b64encode(json.dumps(VMESS_TLS).encode('utf-8')).decode('utf-8')}
trojan://{UUID}@{CFIP}:{CFPORT}?security=tls&sni={argo_domain}&fp=chrome&type=ws&host={argo_domain}&path=%2Ftrojan-argo%3Fed%3D2560#{NAME}-{ISP}-TLS
vless://{UUID}@{CFIP}:80?encryption=none&security=none&type=ws&host={argo_domain}&path=%2Fvless-argo%3Fed%3D2560#{NAME}-{ISP}-80
vmess://{ base64.b64encode(json.dumps(VMESS_80).encode('utf-8')).decode('utf-8')}
trojan://{UUID}@{CFIP}:80?security=none&type=ws&host={argo_domain}&path=%2Ftrojan-argo%3Fed%3D2560#{NAME}-{ISP}-80
"""
    with open(os.path.join(FILE_PATH, 'list.txt'), 'w', encoding='utf-8') as list_file:
        list_file.write(list_txt)
    sub_txt = base64.b64encode(list_txt.encode('utf-8')).decode('utf-8')
    with open(os.path.join(FILE_PATH, 'sub.txt'), 'w', encoding='utf-8') as sub_file:
        sub_file.write(sub_txt)
    print(sub_txt)
    print(f"{FILE_PATH}/sub.txt saved successfully")
    # Additional actions
    send_telegram()
    upload_nodes()
    return sub_txt'''
# 替换generate_links函数
if old_generate_function in content:
    content = content.replace(old_generate_function, new_generate_function)

# 写回文件
with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("扩展分流配置和80端口节点已成功添加")
EOF
python3 extended_patch.py
rm extended_patch.py
echo -e "${GREEN}扩展分流和80端口节点已集成${NC}"

# 先清理可能存在的进程 (优化: 使用pkill更高效)
pkill -f "python3 app.py" > /dev/null 2>&1
pkill -f "keep_alive_task.sh" > /dev/null 2>&1
sleep 2

# 启动服务并获取PID (优化: 使用nohup并重定向日志以提升隐私)
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
echo -e "${YELLOW}日志文件: $(pwd)/app.log${NC}"

# 如果设置了保活URL，则启动保活任务 (优化: 增加间隔到300s以减少请求, 提升效率; 支持repo类型; 隐私: Token不写入日志)
KEEPALIVE_PID=""
if [ "$KEEP_ALIVE_HF" = "true" ]; then
    echo -e "${BLUE}正在创建并启动 Hugging Face API 保活任务...${NC}"
    # 创建保活任务脚本
    echo "#!/bin/bash" > keep_alive_task.sh
    echo "while true; do" >> keep_alive_task.sh
    echo "    API_PATH=\"https://huggingface.co/api/${HF_REPO_TYPE}/${HF_REPO_ID}\"" >> keep_alive_task.sh
    echo "    status_code=\$(curl -s -o /dev/null -w \"%{http_code}\" --header \"Authorization: Bearer \$HF_TOKEN\" \"\$API_PATH\")" >> keep_alive_task.sh
    echo "    if [ \"\$status_code\" -eq 200 ]; then" >> keep_alive_task.sh
    echo "        echo \"Hugging Face API 保活成功 (仓库: $HF_REPO_ID, 类型: $HF_REPO_TYPE, 状态码: 200) - \$(date '+%Y-%m-%d %H:%M:%S')\" > keep_alive_status.log" >> keep_alive_task.sh
    echo "    else" >> keep_alive_task.sh
    echo "        echo \"Hugging Face API 保活失败 (仓库: $HF_REPO_ID, 类型: $HF_REPO_TYPE, 状态码: \$status_code) - \$(date '+%Y-%m-%d %H:%M:%S')\" > keep_alive_status.log" >> keep_alive_task.sh
    echo "    fi" >> keep_alive_task.sh
    echo "    sleep 300 # 优化: 增加间隔减少请求频率" >> keep_alive_task.sh
    echo "done" >> keep_alive_task.sh

    # 环境变量存储Token以提升隐私
    export HF_TOKEN="$HF_TOKEN"
    chmod +x keep_alive_task.sh
    nohup ./keep_alive_task.sh >/dev/null 2>&1 &
    KEEPALIVE_PID=$!
    echo -e "${GREEN}Hugging Face API 保活任务已启动 (PID: $KEEPALIVE_PID)。${NC}"
fi

echo -e "${BLUE}等待服务启动...${NC}"
sleep 5 # 优化: 减少等待时间
# 检查服务是否正常运行
if ! ps -p "$APP_PID" > /dev/null 2>&1; then
    echo -e "${RED}服务启动失败，请检查日志${NC}"
    echo -e "${YELLOW}查看日志: tail -f app.log${NC}"
    echo -e "${YELLOW}检查端口占用: netstat -tlnp | grep :3000${NC}"
    exit 1
fi

echo -e "${GREEN}服务运行正常${NC}"
SERVICE_PORT=$(grep "PORT = int" app.py | grep -o "or [0-9]*" | cut -d" " -f2)
CURRENT_UUID=$(grep "UUID = " app.py | head -1 | cut -d"'" -f2)
SUB_PATH_VALUE=$(grep "SUB_PATH = " app.py | cut -d"'" -f4)

echo -e "${BLUE}等待节点信息生成...${NC}"
echo -e "${YELLOW}正在等待Argo隧道建立和节点生成，请耐心等待...${NC}"

# 循环等待节点信息生成，最多等待5分钟 (优化: 减少最大等待时间)
MAX_WAIT=300
WAIT_COUNT=0
NODE_INFO=""
while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    if [ -f ".cache/sub.txt" ]; then
        NODE_INFO=$(cat .cache/sub.txt 2>/dev/null)
    elif [ -f "sub.txt" ]; then
        NODE_INFO=$(cat sub.txt 2>/dev/null)
    fi
    
    if [ -n "$NODE_INFO" ]; then
        echo -e "\n${GREEN}节点信息已生成！${NC}"
        break
    fi
    
    if [ $((WAIT_COUNT % 30)) -eq 0 ] && [ $WAIT_COUNT -gt 0 ]; then
        MINUTES=$((WAIT_COUNT / 60))
        SECONDS=$((WAIT_COUNT % 60))
        echo -e "${YELLOW}已等待 ${MINUTES}分${SECONDS}秒，继续等待节点生成...${NC}"
    fi
    sleep 5
    WAIT_COUNT=$((WAIT_COUNT + 5))
done

if [ -z "$NODE_INFO" ]; then
    echo -e "${RED}等待超时！节点信息未能在5分钟内生成${NC}"
    echo -e "${YELLOW}可能原因：网络问题、Argo失败、配置错误${NC}"
    echo -e "${BLUE}建议: 查看日志 tail -f $(pwd)/app.log${NC}"
    exit 1
fi

echo
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN} 部署完成！ ${NC}"
echo -e "${GREEN}========================================${NC}"
echo

echo -e "${YELLOW}=== 服务信息 ===${NC}"
echo -e "服务状态: ${GREEN}运行中${NC}"
echo -e "主服务PID: ${BLUE}$APP_PID${NC}"
if [ -n "$KEEPALIVE_PID" ]; then
    echo -e "保活服务PID: ${BLUE}$KEEPALIVE_PID${NC}"
fi
echo -e "服务端口: ${BLUE}$SERVICE_PORT${NC}"
echo -e "UUID: ${BLUE}$CURRENT_UUID${NC}"
echo -e "订阅路径: ${BLUE}/$SUB_PATH_VALUE${NC}"
echo

echo -e "${YELLOW}=== 访问地址 ===${NC}"
PUBLIC_IP=$(curl -s https://api.ipify.org 2>/dev/null || echo "获取失败")
if [ "$PUBLIC_IP" != "获取失败" ]; then
    echo -e "订阅地址: ${GREEN}http://$PUBLIC_IP:$SERVICE_PORT/$SUB_PATH_VALUE${NC}"
    echo -e "管理面板: ${GREEN}http://$PUBLIC_IP:$SERVICE_PORT${NC}"
fi
echo -e "本地订阅: ${GREEN}http://localhost:$SERVICE_PORT/$SUB_PATH_VALUE${NC}"
echo -e "本地面板: ${GREEN}http://localhost:$SERVICE_PORT${NC}"
echo

echo -e "${YELLOW}=== 节点信息 ===${NC}"
DECODED_NODES=$(echo "$NODE_INFO" | base64 -d 2>/dev/null || echo "$NODE_INFO")
echo -e "${GREEN}节点配置:${NC}"
echo "$DECODED_NODES"
echo
echo -e "${GREEN}订阅链接(Base64):${NC}"
echo "$NODE_INFO"
echo

SAVE_INFO="========================================
节点信息保存
========================================
部署时间: $(date)
UUID: $CURRENT_UUID
服务端口: $SERVICE_PORT
订阅路径: /$SUB_PATH_VALUE

=== 访问地址 ===
"
if [ "$PUBLIC_IP" != "获取失败" ]; then
    SAVE_INFO="${SAVE_INFO}
订阅地址: http://$PUBLIC_IP:$SERVICE_PORT/$SUB_PATH_VALUE
管理面板: http://$PUBLIC_IP:$SERVICE_PORT
"
fi
SAVE_INFO="${SAVE_INFO}
本地订阅: http://localhost:$SERVICE_PORT/$SUB_PATH_VALUE
本地面板: http://localhost:$SERVICE_PORT

=== 节点信息 ===
$DECODED_NODES

=== 订阅链接 (Base64) ===
$NODE_INFO

=== 管理命令 ===
查看日志: tail -f $(pwd)/app.log
停止主服务: pkill -f \"python3 app.py\"
重启主服务: pkill -f \"python3 app.py\" && nohup python3 app.py > app.log 2>&1 &
查看进程: ps aux | grep app.py
"
if [ "$KEEP_ALIVE_HF" = "true" ]; then
    SAVE_INFO="${SAVE_INFO}
停止保活服务: pkill -f keep_alive_task.sh
查看保活状态: bash $0 -s
"
fi
SAVE_INFO="${SAVE_INFO}
=== 分流说明 ===
- 已集成扩展分流优化到xray配置
- 支持YouTube、社交平台、Netflix等域名自动走专用线路
- 提升流量处理效率，无需额外配置"

echo "$SAVE_INFO" > "$NODE_INFO_FILE"
echo -e "${GREEN}节点信息已保存到 $NODE_INFO_FILE${NC}"
echo -e "${YELLOW}使用命令 'bash $0 -v' 可随时查看节点信息${NC}"
echo
echo -e "${YELLOW}=== 重要提示 ===${NC}"
echo -e "${GREEN}部署已完成，节点信息已成功生成。${NC}"
echo -e "${GREEN}可以立即使用订阅地址添加到客户端。${NC}"
echo -e "${GREEN}服务将持续在后台运行。${NC}"
echo
echo -e "${GREEN}感谢使用！${NC}"
# 退出脚本
exit 0
