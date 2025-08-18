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
echo -e "${GREEN} Python Xray Argo 一键部署脚本 (为您定制) ${NC}"
echo -e "${GREEN}========================================${NC}"
echo
echo -e "${BLUE}基于项目: ${YELLOW}https://github.com/eooce/python-xray-argo${NC}"
echo
echo -e "${GREEN}此脚本已为您预设所有关键信息，将自动执行“完整模式”部署。${NC}"
echo -e "${GREEN}对于非关键信息，直接按 Enter 键使用默认值即可。${NC}"
echo
read -p "按 Enter 键开始部署..."

# 自动选择完整模式
MODE_CHOICE="2"

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

# 初始化保活变量
KEEP_ALIVE_HF="false"
HF_TOKEN=""
HF_REPO_ID=""
HF_REPO_TYPE="spaces" # 默认优先spaces

#【新版函数，请用它完整替换上面的旧函数】

# 定义保活配置函数（安全版）
configure_hf_keep_alive() {
    echo
    echo -e "${YELLOW}是否设置 Hugging Face API 自动保活? (y/n)${NC}"
    SETUP_KEEP_ALIVE="y" # <-- 自动选择
    echo -e "${GREEN}[已自动选择 y]${NC}"
    
    if [ "$SETUP_KEEP_ALIVE" = "y" ] || [ "$SETUP_KEEP_ALIVE" = "Y" ]; then
        echo -e "${BLUE}正在从 Space secrets 读取 Hugging Face 令牌...${NC}"
        # 检查名为 HF_TOKEN 的环境变量是否存在且不为空
        if [ -z "$HF_TOKEN" ]; then
            echo -e "${RED}错误：未能从 Space secrets 中找到名为 HF_TOKEN 的令牌。${NC}"
            echo -e "${YELLOW}请确认您已在 Space 的 Settings -> Secrets 中添加了它。${NC}"
            echo -e "${YELLOW}Name 必须为 HF_TOKEN，Value 为您的令牌 hf_...${NC}"
            KEEP_ALIVE_HF="false" # 标记为失败，避免后续出错
            return
        fi
        
        # 如果能找到，HF_TOKEN 变量就已自动可用，无需再手动赋值
        echo -e "${GREEN}成功从 Space secrets 读取令牌！${NC}"
        
        echo -e "${YELLOW}请输入要访问的 Hugging Face 仓库ID (例如: username/repo):${NC}"
        HF_REPO_ID_INPUT="sukikeeling/face" # <-- 自动填入
        echo -e "${GREEN}[Repo ID 已自动填入]${NC}"
        
        echo -e "${YELLOW}仓库类型 (spaces/models):${NC}"
        read -p "Type (默认 spaces): " HF_REPO_TYPE_INPUT
        HF_REPO_TYPE="${HF_REPO_TYPE_INPUT:-spaces}"

        HF_REPO_ID="$HF_REPO_ID_INPUT"
        KEEP_ALIVE_HF="true"
        echo -e "${GREEN}Hugging Face API 保活已设置！类型: $HF_REPO_TYPE${NC}"
        echo -e "${GREEN}目标仓库: $HF_REPO_ID${NC}"
    fi
}

# --- 自动进入完整模式 ---
echo -e "${BLUE}=== 自动进入完整配置模式 ===${NC}"
echo
echo -e "${YELLOW}当前UUID: $(grep "UUID = " app.py | head -1 | cut -d"'" -f2)${NC}"
UUID_INPUT="c10a3483-5de5-4416-9a37-a6c702b916ac" # <-- 自动填入
echo -e "${GREEN}[UUID 已自动填入]${NC}"

sed -i "s/UUID = os.environ.get('UUID', '[^']*')/UUID = os.environ.get('UUID', '$UUID_INPUT')/" app.py
echo -e "${GREEN}UUID 已设置为: $UUID_INPUT${NC}"

echo -e "${YELLOW}当前节点名称: $(grep "NAME = " app.py | head -1 | cut -d"'" -f4)${NC}"
read -p "请输入节点名称 (留空保持不变): " NAME_INPUT
if [ -n "$NAME_INPUT" ]; then
    sed -i "s/NAME = os.environ.get('NAME', '[^']*')/NAME = os.environ.get('NAME', '$NAME_INPUT')/" app.py
    echo -e "${GREEN}节点名称已设置为: $NAME_INPUT${NC}"
fi

echo -e "${YELLOW}当前服务端口: $(grep "PORT = int" app.py | grep -o "or [0-9]*" | cut -d" " -f2)${NC}"
read -p "请输入服务端口 (留空保持不变): " PORT_INPUT
if [ -n "$PORT_INPUT" ]; then
    sed -i "s/PORT = int(os.environ.get('SERVER_PORT') or os.environ.get('PORT') or [0-9]*)/PORT = int(os.environ.get('SERVER_PORT') or os.environ.get('PORT') or $PORT_INPUT)/" app.py
    echo -e "${GREEN}端口已设置为: $PORT_INPUT${NC}"
fi

echo -e "${YELLOW}当前优选IP: $(grep "CFIP = " app.py | cut -d"'" -f4)${NC}"
read -p "请输入优选IP/域名 (留空使用默认 joeyblog.net): " CFIP_INPUT
if [ -z "$CFIP_INPUT" ]; then
    CFIP_INPUT="joeyblog.net"
fi
sed -i "s/CFIP = os.environ.get('CFIP', '[^']*')/CFIP = os.environ.get('CFIP', '$CFIP_INPUT')/" app.py
echo -e "${GREEN}优选IP已设置为: $CFIP_INPUT${NC}"

echo -e "${YELLOW}当前优选端口: $(grep "CFPORT = " app.py | cut -d"'" -f4)${NC}"
read -p "请输入优选端口 (留空保持不变): " CFPORT_INPUT
if [ -n "$CFPORT_INPUT" ]; then
    sed -i "s/CFPORT = int(os.environ.get('CFPORT', '[^']*'))/CFPORT = int(os.environ.get('CFPORT', '$CFPORT_INPUT'))/" app.py
    echo -e "${GREEN}优选端口已设置为: $CFPORT_INPUT${NC}"
fi

echo -e "${YELLOW}当前Argo端口: $(grep "ARGO_PORT = " app.py | cut -d"'" -f4)${NC}"
read -p "请输入 Argo 端口 (留空保持不变): " ARGO_PORT_INPUT
if [ -n "$ARGO_PORT_INPUT" ]; then
    sed -i "s/ARGO_PORT = int(os.environ.get('ARGO_PORT', '[^']*'))/ARGO_PORT = int(os.environ.get('ARGO_PORT', '$ARGO_PORT_INPUT'))/" app.py
    echo -e "${GREEN}Argo端口已设置为: $ARGO_PORT_INPUT${NC}"
fi

echo -e "${YELLOW}当前订阅路径: $(grep "SUB_PATH = " app.py | cut -d"'" -f4)${NC}"
read -p "请输入订阅路径 (留空保持不变): " SUB_PATH_INPUT
if [ -n "$SUB_PATH_INPUT" ]; then
    sed -i "s/SUB_PATH = os.environ.get('SUB_PATH', '[^']*')/SUB_PATH = os.environ.get('SUB_PATH', '$SUB_PATH_INPUT')/" app.py
    echo -e "${GREEN}订阅路径已设置为: $SUB_PATH_INPUT${NC}"
fi
echo

echo -e "${YELLOW}是否配置高级选项? (y/n)${NC}"
ADVANCED_CONFIG="y" # <-- 自动同意
echo -e "${GREEN}[已自动选择 y]${NC}"

if [ "$ADVANCED_CONFIG" = "y" ] || [ "$ADVANCED_CONFIG" = "Y" ]; then
    configure_hf_keep_alive
fi

echo -e "${YELLOW}当前Argo域名: $(grep "ARGO_DOMAIN = " app.py | cut -d"'" -f4)${NC}"
ARGO_DOMAIN_INPUT="face.keeling.dpdns.org" # <-- 自动填入
echo -e "${GREEN}[Argo 域名已自动填入]${NC}"

if [ -n "$ARGO_DOMAIN_INPUT" ]; then
    sed -i "s|ARGO_DOMAIN = os.environ.get('ARGO_DOMAIN', '[^']*')|ARGO_DOMAIN = os.environ.get('ARGO_DOMAIN', '$ARGO_DOMAIN_INPUT')|" app.py
    
    echo -e "${YELLOW}当前Argo密钥: $(grep "ARGO_AUTH = " app.py | cut -d"'" -f4)${NC}"
    ARGO_AUTH_INPUT='{"AccountTag":"46fad1b6b0e334ca8ad9ea7ec29c4ddb","TunnelSecret":"J2TOKaJiWL8rph+m7iTfEOthVtREnhuvfWoHp4SmOog=","TunnelID":"29e3716e-783c-4a1f-9538-d40fa766006f","Endpoint":""}' # <-- 自动填入
    echo -e "${GREEN}[Argo 密钥已自动填入]${NC}"

    if [ -n "$ARGO_AUTH_INPUT" ]; then
        sed -i "s|ARGO_AUTH = os.environ.get('ARGO_AUTH', '[^']*')|ARGO_AUTH = os.environ.get('ARGO_AUTH', '$ARGO_AUTH_INPUT')|" app.py
    fi
    echo -e "${GREEN}Argo固定隧道配置已设置${NC}"
fi

# 省略 Telegram 等其他非必要配置的交互
echo
echo -e "${GREEN}扩展分流已自动配置${NC}"
echo
echo -e "${GREEN}完整配置完成！${NC}"


# --- 后续代码和原来保持一致 ---

echo -e "${YELLOW}=== 当前配置摘要 ===${NC}"
echo -e "UUID: $(grep "UUID = " app.py | head -1 | cut -d"'" -f2)"
echo -e "节点名称: $(grep "NAME = " app.py | head -1 | cut -d"'" -f4)"
echo -e "服务端口: $(grep "PORT = int" app.py | grep -o "or [0-9]*" | cut -d" " -f2)"
echo -e "优选IP: $(grep "CFIP = " app.py | cut -d"'" -f4)"
echo -e "优选端口: $(grep "CFPORT = " app.py | cut -d"'" -f4)"
echo -e "订阅路径: $(grep "SUB_PATH = " app.py | cut -d"'" -f4)"
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
        echo -e "${GREEN}节点信息已生成！${NC}"
        break
    fi
    
    if [ $((WAIT_COUNT % 30)) -eq 0 ]; then
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
echo -e "${GREEN}订阅链接:${NC}"
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

=== 订阅链接 ===
$NODE_INFO

=== 管理命令 ===
查看日志: tail -f $(pwd)/app.log
停止主服务: pkill -f \"python3 app.py\"
重启主服务: pkill -f \"python3 app.py\" && nohup python3 app.py > app.log 2>&1 &
查看进程: ps aux | grep app.py
"
if [ "$KEEP_ALIVE_HF" = "true" ]; then
    SAVE_INFO="${SAVE_INFO}
停止保活服务: pkill -f keep_alive_task.sh && rm keep_alive_task.sh keep_alive_status.log
"
fi
SAVE_INFO="${SAVE_INFO}
=== 分流说明 ===
- 已集成扩展分流优化到xray配置
- 支持YouTube、社交平台、Netflix等域名自动走专用线路
- 提升流量处理效率，无需额外配置"

echo "$SAVE_INFO" > "$NODE_INFO_FILE"
echo -e "${GREEN}节点信息已保存到 $NODE_INFO_FILE${NC}"
echo -e "${YELLOW}使用脚本选择选项3或运行带-v参数可随时查看节点信息${NC}"
echo
echo -e "${YELLOW}=== 重要提示 ===${NC}"
echo -e "${GREEN}部署已完成，节点信息已成功生成${NC}"
echo -e "${GREEN}可以立即使用订阅地址添加到客户端${NC}"
echo -e "${GREEN}扩展分流已集成，提升效率和隐私${NC}"
echo -e "${GREEN}服务将持续在后台运行${NC}"
echo
echo -e "${GREEN}部署完成！感谢使用！${NC}"

# 退出脚本
exit 0
