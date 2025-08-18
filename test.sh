#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

NODE_INFO_FILE="$HOME/.xray_nodes_info.enc"
PROJECT_DIR_NAME="python-xray-argo"
RULES_FILE="rules.json"
ENCRYPT_KEY=""
KEEP_ALIVE_INTERVAL=300

# 检查openssl用于加密
if ! command -v openssl &> /dev/null; then
    sudo apt-get update -qq && sudo apt-get install -y openssl
fi

# UUID生成（优化版）
generate_uuid() {
    if command -v uuidgen &> /dev/null; then
        uuidgen | tr '[:upper:]' '[:lower:]'
    elif command -v python3 &> /dev/null; then
        python3 -c "import uuid; print(str(uuid.uuid4()))"
    else
        openssl rand -hex 16 | sed 's/\(........\)\(....\)\(....\)\(....\)\(............\)/\1-\2-\3-\4-\5/' | tr '[:upper:]' '[:lower:]'
    fi
}

# 加密文件
encrypt_file() {
    local input_file=$1
    local output_file=$2
    openssl enc -aes-256-cbc -salt -in "$input_file" -out "$output_file" -k "$ENCRYPT_KEY" 2>/dev/null
}

# 解密文件
decrypt_file() {
    local input_file=$1
    local output_file=$2
    openssl enc -aes-256-cbc -d -in "$input_file" -out "$output_file" -k "$ENCRYPT_KEY" 2>/dev/null
}

# 查看节点信息
if [ "$1" = "-v" ]; then
    if [ -f "$NODE_INFO_FILE" ]; then
        echo -e "${YELLOW}请输入解密密码:${NC}"
        read -sp "密码: " ENCRYPT_KEY
        echo
        decrypt_file "$NODE_INFO_FILE" /tmp/node_info.dec
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}========================================${NC}"
            echo -e "${GREEN} 节点信息查看 ${NC}"
            echo -e "${GREEN}========================================${NC}"
            echo
            cat /tmp/node_info.dec
            echo
            rm /tmp/node_info.dec
        else
            echo -e "${RED}解密失败，密码错误或文件损坏${NC}"
        fi
    else
        echo -e "${RED}未找到节点信息文件${NC}"
        echo -e "${YELLOW}请先运行部署脚本生成节点信息${NC}"
    fi
    exit 0
fi

clear
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN} Python Xray Argo 一键部署脚本 (魔改版) ${NC}"
echo -e "${GREEN}========================================${NC}"
echo
echo -e "${BLUE}基于项目: ${YELLOW}https://github.com/eooce/python-xray-argo${NC}"
echo -e "${BLUE}脚本仓库: ${YELLOW}https://github.com/byJoey/free-vps-py${NC}"
echo -e "${BLUE}TG交流群: ${YELLOW}https://t.me/+ft-zI76oovgwNmRh${NC}"
echo -e "${RED}脚本作者YouTube: ${YELLOW}https://www.youtube.com/@joeyblog${RED}"
echo
echo -e "${GREEN}魔改特性: Shadowsocks支持、加密存储、动态分流、自动诊断${NC}"
echo -e "${GREEN}效率: 异步安装、自适应保活、动态等待${NC}"
echo -e "${GREEN}隐私: 节点信息加密、日志加密、匿名化配置${NC}"
echo
echo -e "${YELLOW}请输入节点信息加密密码:${NC}"
read -sp "密码: " ENCRYPT_KEY
echo
if [ -z "$ENCRYPT_KEY" ]; then
    echo -e "${RED}加密密码不能为空${NC}"
    exit 1
fi
echo -e "${YELLOW}请选择操作:${NC}"
echo -e "${BLUE}1) 极速模式 - 只修改UUID并启动${NC}"
echo -e "${BLUE}2) 完整模式 - 详细配置所有选项${NC}"
echo -e "${BLUE}3) 查看节点信息 - 显示已保存的节点信息${NC}"
echo -e "${BLUE}4) 查看保活状态 - 检查Hugging Face API保活状态${NC}"
echo
read -p "请输入选择 (1/2/3/4): " MODE_CHOICE

if [ "$MODE_CHOICE" = "3" ]; then
    if [ -f "$NODE_INFO_FILE" ]; then
        echo -e "${YELLOW}请输入解密密码:${NC}"
        read -sp "密码: " ENCRYPT_KEY
        echo
        decrypt_file "$NODE_INFO_FILE" /tmp/node_info.dec
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}========================================${NC}"
            echo -e "${GREEN} 节点信息查看 ${NC}"
            echo -e "${GREEN}========================================${NC}"
            echo
            cat /tmp/node_info.dec
            echo
            rm /tmp/node_info.dec
            echo -e "${YELLOW}提示: 如需重新部署，请重新运行脚本选择模式1或2${NC}"
        else
            echo -e "${RED}解密失败，密码错误或文件损坏${NC}"
        fi
    else
        echo -e "${RED}未找到节点信息文件${NC}"
        echo -e "${YELLOW}请先运行部署脚本生成节点信息${NC}"
        echo -e "${BLUE}是否现在开始部署? (y/n)${NC}"
        read -p "> " START_DEPLOY
        if [ "$START_DEPLOY" = "y" ] || [ "$START_DEPLOY" = "Y" ]; then
            echo -e "${YELLOW}请输入加密密码:${NC}"
            read -sp "密码: " ENCRYPT_KEY
            echo
            if [ -z "$ENCRYPT_KEY" ]; then
                echo -e "${RED}加密密码不能为空${NC}"
                exit 1
            fi
            echo -e "${YELLOW}请选择部署模式:${NC}"
            echo -e "${BLUE}1) 极速模式${NC}"
            echo -e "${BLUE}2) 完整模式${NC}"
            read -p "请输入选择 (1/2): " MODE_CHOICE
        else
            echo -e "${GREEN}退出脚本${NC}"
            exit 0
        fi
    fi
    if [ "$MODE_CHOICE" != "1" ] && [ "$MODE_CHOICE" != "2" ]; then
        echo -e "${GREEN}退出脚本${NC}"
        exit 0
    fi
fi

if [ "$MODE_CHOICE" = "4" ]; then
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
        if [ -f "keep_alive_status.log.enc" ]; then
            decrypt_file "keep_alive_status.log.enc" /tmp/keep_alive_status.log
            echo -e "目标仓库: ${YELLOW}$(grep '仓库:' /tmp/keep_alive_status.log | tail -1 | cut -d' ' -f3-)${NC}"
            echo -e "\n${YELLOW}--- 最近一次保活状态 ---${NC}"
            cat /tmp/keep_alive_status.log
            rm /tmp/keep_alive_status.log
        else
            echo -e "${YELLOW}尚未生成状态日志，请稍等片刻(最多5分钟)后重试...${NC}"
        fi
    else
        echo -e "服务状态: ${RED}未运行${NC}"
        echo -e "${YELLOW}提示: 您可能尚未部署服务或未设置保活。${NC}"
    fi
    echo
    exit 0
fi

echo -e "${BLUE}检查网络连通性...${NC}"
ping -c 1 api.cloudflare.com > /dev/null 2>&1 || {
    echo -e "${RED}无法连接Cloudflare，请检查网络${NC}"
    exit 1
}
ping -c 1 huggingface.co > /dev/null 2>&1 || {
    echo -e "${RED}无法连接Hugging Face，请检查网络${NC}"
    exit 1
}

echo -e "${BLUE}检查并安装依赖...${NC}"
sudo apt-get update -qq || true
DEPS="python3 python3-pip git unzip openssl"
for dep in $DEPS; do
    if ! command -v $dep &> /dev/null && [ "$dep" != "python3-pip" ]; then
        echo -e "${YELLOW}正在安装 $dep...${NC}"
        sudo apt-get install -y $dep &
    fi
done
wait
if ! python3 -c "import requests" &> /dev/null; then
    echo -e "${YELLOW}正在安装 Python 依赖: requests...${NC}"
    pip3 install --user requests
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

# 初始化变量
KEEP_ALIVE_HF="false"
HF_TOKEN=""
HF_REPO_ID=""
HF_REPO_TYPE="spaces"
SS_PASSWORD=$(openssl rand -base64 12)
PROTOCOLS="vless,vmess,trojan"

# 保活配置
configure_hf_keep_alive() {
    echo -e "${YELLOW}是否设置 Hugging Face API 自动保活? (y/n)${NC}"
    read -p "> " SETUP_KEEP_ALIVE
    if [ "$SETUP_KEEP_ALIVE" = "y" ] || [ "$SETUP_KEEP_ALIVE" = "Y" ]; then
        echo -e "${YELLOW}请输入您的 Hugging Face 访问令牌 (Token):${NC}"
        read -sp "Token: " HF_TOKEN_INPUT
        echo
        if [ -z "$HF_TOKEN_INPUT" ]; then
            echo -e "${RED}错误：Token 不能为空。已取消保活设置。${NC}"
            return
        fi
        echo -e "${YELLOW}请输入要访问的 Hugging Face 仓库ID (例如: username/repo):${NC}"
        read -p "Repo ID: " HF_REPO_ID_INPUT
        if [ -z "$HF_REPO_ID_INPUT" ]; then
            echo -e "${RED}错误：仓库ID 不能为空。已取消保活设置。${NC}"
            return
        fi
        echo -e "${YELLOW}仓库类型 (spaces/models):${NC}"
        read -p "Type (默认 spaces): " HF_REPO_TYPE_INPUT
        HF_REPO_TYPE="${HF_REPO_TYPE_INPUT:-spaces}"
        HF_TOKEN="$HF_TOKEN_INPUT"
        HF_REPO_ID="$HF_REPO_ID_INPUT"
        KEEP_ALIVE_HF="true"
        echo -e "${GREEN}Hugging Face API 保活已设置！类型: $HF_REPO_TYPE${NC}"
        echo -e "${GREEN}目标仓库: $HF_REPO_ID${NC}"
    fi
}

# 分流规则文件
cat > "$RULES_FILE" << 'EOF'
{
    "media": [
        "youtube.com", "youtu.be", "googlevideo.com", "ytimg.com",
        "gstatic.com", "googleapis.com", "ggpht.com", "googleusercontent.com",
        "facebook.com", "fb.com", "fbcdn.net", "instagram.com", "cdninstagram.com",
        "twitter.com", "x.com", "twimg.com", "t.co",
        "discord.com", "discordapp.com", "discord.gg", "discord.media",
        "telegram.org", "t.me", "telegram.me", "web.telegram.org",
        "whatsapp.com", "whatsapp.net", "meta.com", "meta.ai",
        "tiktok.com", "tiktokv.com", "ttlivecdn.com", "byteoversea.com",
        "netflix.com", "netflix.net", "nflxvideo.net", "nflximg.net",
        "twitch.tv", "ttvnw.net", "jtvnw.net",
        "bilibili.com", "bilibili.tv", "biliapi.net",
        "douyin.com", "douyincdn.com",
        "steampowered.com", "steamstatic.com", "epicgames.com"
    ]
}
EOF

if [ "$MODE_CHOICE" = "1" ]; then
    echo -e "${BLUE}=== 极速模式 ===${NC}"
    echo
    UUID_INPUT=$(generate_uuid)
    echo -e "${GREEN}自动生成UUID: $UUID_INPUT${NC}"
    sed -i "s/UUID = os.environ.get('UUID', '[^']*')/UUID = os.environ.get('UUID', '$UUID_INPUT')/" app.py
    echo -e "${GREEN}UUID 已设置为: $UUID_INPUT${NC}"
    sed -i "s/CFIP = os.environ.get('CFIP', '[^']*')/CFIP = os.environ.get('CFIP', 'joeyblog.net')/" app.py
    echo -e "${GREEN}优选IP已自动设置为: joeyblog.net${NC}"
    configure_hf_keep_alive
    echo -e "${GREEN}扩展分流已自动配置${NC}"
else
    echo -e "${BLUE}=== 完整配置模式 ===${NC}"
    echo
    echo -e "${YELLOW}当前UUID: $(grep "UUID = " app.py | head -1 | cut -d"'" -f2)${NC}"
    read -p "请输入新的 UUID (留空自动生成): " UUID_INPUT
    if [ -z "$UUID_INPUT" ]; then
        UUID_INPUT=$(generate_uuid)
        echo -e "${GREEN}自动生成UUID: $UUID_INPUT${NC}"
    fi
    sed -i "s/UUID = os.environ.get('UUID', '[^']*')/UUID = os.environ.get('UUID', '$UUID_INPUT')/" app.py
    echo -e "${GREEN}UUID 已设置为: $UUID_INPUT${NC}"
    
    echo -e "${YELLOW}是否随机生成节点名称和订阅路径? (y/n)${NC}"
    read -p "> " RANDOMIZE
    if [ "$RANDOMIZE" = "y" ] || [ "$RANDOMIZE" = "Y" ]; then
        NAME_INPUT=$(openssl rand -hex 8)
        SUB_PATH_INPUT=$(openssl rand -hex 8)
        echo -e "${GREEN}随机节点名称: $NAME_INPUT${NC}"
        echo -e "${GREEN}随机订阅路径: /$SUB_PATH_INPUT${NC}"
    else
        echo -e "${YELLOW}当前节点名称: $(grep "NAME = " app.py | head -1 | cut -d"'" -f4)${NC}"
        read -p "请输入节点名称 (留空保持不变): " NAME_INPUT
        echo -e "${YELLOW}当前订阅路径: $(grep "SUB_PATH = " app.py | cut -d"'" -f4)${NC}"
        read -p "请输入订阅路径 (留空保持不变): " SUB_PATH_INPUT
    fi
    if [ -n "$NAME_INPUT" ]; then
        sed -i "s/NAME = os.environ.get('NAME', '[^']*')/NAME = os.environ.get('NAME', '$NAME_INPUT')/" app.py
        echo -e "${GREEN}节点名称已设置为: $NAME_INPUT${NC}"
    fi
    if [ -n "$SUB_PATH_INPUT" ]; then
        sed -i "s/SUB_PATH = os.environ.get('SUB_PATH', '[^']*')/SUB_PATH = os.environ.get('SUB_PATH', '$SUB_PATH_INPUT')/" app.py
        echo -e "${GREEN}订阅路径已设置为: $SUB_PATH_INPUT${NC}"
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
    
    echo -e "${YELLOW}选择启用协议 (vless,vmess,trojan,shadowsocks，用逗号分隔):${NC}"
    read -p "协议 (默认 vless,vmess,trojan): " PROTOCOLS_INPUT
    PROTOCOLS="${PROTOCOLS_INPUT:-vless,vmess,trojan}"
    echo -e "${GREEN}已启用协议: $PROTOCOLS${NC}"
    
    configure_hf_keep_alive
fi

echo -e "${YELLOW}=== 当前配置摘要 ===${NC}"
echo -e "UUID: $(grep "UUID = " app.py | head -1 | cut -d"'" -f2)"
echo -e "节点名称: $(grep "NAME = " app.py | head -1 | cut -d"'" -f4)"
echo -e "服务端口: $(grep "PORT = int" app.py | grep -o "or [0-9]*" | cut -d" " -f2)"
echo -e "优选IP: $(grep "CFIP = " app.py | cut -d"'" -f4)"
echo -e "订阅路径: $(grep "SUB_PATH = " app.py | cut -d"'" -f4)"
if [ "$KEEP_ALIVE_HF" = "true" ]; then
    echo -e "保活仓库: $HF_REPO_ID ($HF_REPO_TYPE)"
fi
echo -e "${YELLOW}启用协议: $PROTOCOLS${NC}"
echo -e "${YELLOW}========================${NC}"

echo -e "${BLUE}正在应用扩展配置...${NC}"
cat > extended_patch.py << 'EOF'
# coding: utf-8
import os, base64, json, subprocess, time
# 读取app.py和rules.json
with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()
with open('rules.json', 'r', encoding='utf-8') as f:
    rules = json.load(f)
# 动态生成inbounds
inbounds = [
    {
        "port": "ARGO_PORT",
        "protocol": "vless",
        "settings": {
            "clients": [{"id": "UUID", "flow": "xtls-rprx-vision"}],
            "decryption": "none",
            "fallbacks": [
                {"dest": 3001},
                {"path": "/vless-argo", "dest": 3002},
                {"path": "/vmess-argo", "dest": 3003},
                {"path": "/trojan-argo", "dest": 3004},
                {"path": "/ss-argo", "dest": 3005}
            ]
        },
        "streamSettings": {"network": "tcp"}
    },
    {
        "port": 3001,
        "listen": "127.0.0.1",
        "protocol": "vless",
        "settings": {
            "clients": [{"id": "UUID"}],
            "decryption": "none"
        },
        "streamSettings": {"network": "ws", "security": "none"}
    }
]
if "vless" in os.environ.get("PROTOCOLS", "").split(","):
    inbounds.append({
        "port": 3002,
        "listen": "127.0.0.1",
        "protocol": "vless",
        "settings": {
            "clients": [{"id": "UUID", "level": 0}],
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
    })
if "vmess" in os.environ.get("PROTOCOLS", "").split(","):
    inbounds.append({
        "port": 3003,
        "listen": "127.0.0.1",
        "protocol": "vmess",
        "settings": {
            "clients": [{"id": "UUID", "alterId": 0}]
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
    })
if "trojan" in os.environ.get("PROTOCOLS", "").split(","):
    inbounds.append({
        "port": 3004,
        "listen": "127.0.0.1",
        "protocol": "trojan",
        "settings": {
            "clients": [{"password": "UUID"}]
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
    })
if "shadowsocks" in os.environ.get("PROTOCOLS", "").split(","):
    inbounds.append({
        "port": 3005,
        "listen": "127.0.0.1",
        "protocol": "shadowsocks",
        "settings": {
            "clients": [{"password": os.environ.get("SS_PASSWORD", "default_password"), "method": "aes-256-gcm"}]
        },
        "streamSettings": {
            "network": "ws",
            "security": "none",
            "wsSettings": {"path": "/ss-argo"}
        },
        "sniffing": {
            "enabled": True,
            "destOverride": ["http", "tls", "quic"],
            "metadataOnly": False
        }
    })
# 配置
new_config = {
    "log": {"access": "/dev/null", "error": "/dev/null", "loglevel": "warning"},
    "inbounds": inbounds,
    "outbounds": [
        {"protocol": "freedom", "tag": "direct"},
        {
            "protocol": "vmess",
            "tag": "media",
            "settings": {
                "vnext": [
                    {
                        "address": "172.233.171.224",
                        "port": 16416,
                        "users": [{"id": "8c1b9bea-cb51-43bb-a65c-0af31bbbf145", "alterId": 0}]
                    },
                    {
                        "address": "104.21.94.34",
                        "port": 16416,
                        "users": [{"id": "8c1b9bea-cb51-43bb-a65c-0af31bbbf145", "alterId": 0}]
                    }
                ]
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
                "domain": rules["media"],
                "outboundTag": "media"
            }
        ]
    }
}
# 替换配置
old_config = 'config ={"log":{"access":"/dev/null","error":"/dev/null","loglevel":"none",}'
content = content[:content.find('config =')] + f'config = {json.dumps(new_config, indent=4)}' + content[content.find('outbounds'):]
# 修改generate_links函数
old_generate_function = '''# Generate links and subscription content
async def generate_links(argo_domain):'''
new_generate_function = '''# Generate links and subscription content
async def generate_links(argo_domain):
    meta_info = subprocess.run(['curl', '-s', 'https://speed.cloudflare.com/meta'], capture_output=True, text=True)
    meta_info = meta_info.stdout.split('"')
    ISP = f"{meta_info[25]}-{meta_info[17]}".replace(' ', '_').strip()
    time.sleep(1)
    list_txt = ""
    if "vless" in os.environ.get("PROTOCOLS", "").split(","):
        list_txt += f"vless://{UUID}@{CFIP}:{CFPORT}?encryption=none&security=tls&sni={argo_domain}&fp=chrome&type=ws&host={argo_domain}&path=%2Fvless-argo%3Fed%3D2560#{NAME}-{ISP}-TLS\\n"
        list_txt += f"vless://{UUID}@{CFIP}:80?encryption=none&security=none&type=ws&host={argo_domain}&path=%2Fvless-argo%3Fed%3D2560#{NAME}-{ISP}-80\\n"
    if "vmess" in os.environ.get("PROTOCOLS", "").split(","):
        VMESS_TLS = {"v": "2", "ps": f"{NAME}-{ISP}-TLS", "add": CFIP, "port": CFPORT, "id": UUID, "aid": "0", "scy": "none", "net": "ws", "type": "none", "host": argo_domain, "path": "/vmess-argo?ed=2560", "tls": "tls", "sni": argo_domain, "alpn": "", "fp": "chrome"}
        VMESS_80 = {"v": "2", "ps": f"{NAME}-{ISP}-80", "add": CFIP, "port": "80", "id": UUID, "aid": "0", "scy": "none", "net": "ws", "type": "none", "host": argo_domain, "path": "/vmess-argo?ed=2560", "tls": "", "sni": "", "alpn": "", "fp": ""}
        list_txt += f"vmess://{ base64.b64encode(json.dumps(VMESS_TLS).encode('utf-8')).decode('utf-8')}\\n"
        list_txt += f"vmess://{ base64.b64encode(json.dumps(VMESS_80).encode('utf-8')).decode('utf-8')}\\n"
    if "trojan" in os.environ.get("PROTOCOLS", "").split(","):
        list_txt += f"trojan://{UUID}@{CFIP}:{CFPORT}?security=tls&sni={argo_domain}&fp=chrome&type=ws&host={argo_domain}&path=%2Ftrojan-argo%3Fed%3D2560#{NAME}-{ISP}-TLS\\n"
        list_txt += f"trojan://{UUID}@{CFIP}:80?security=none&type=ws&host={argo_domain}&path=%2Ftrojan-argo%3Fed%3D2560#{NAME}-{ISP}-80\\n"
    if "shadowsocks" in os.environ.get("PROTOCOLS", "").split(","):
        SS_ENCODED = base64.b64encode(f"aes-256-gcm:{os.environ.get('SS_PASSWORD')}@{CFIP}:{CFPORT}".encode('utf-8')).decode('utf-8')
        list_txt += f"ss://{SS_ENCODED}#{NAME}-{ISP}-SS\\n"
    with open(os.path.join(FILE_PATH, 'list.txt'), 'w', encoding='utf-8') as list_file:
        list_file.write(list_txt)
    sub_txt = base64.b64encode(list_txt.encode('utf-8')).decode('utf-8')
    with open(os.path.join(FILE_PATH, 'sub.txt'), 'w', encoding='utf-8') as sub_file:
        sub_file.write(sub_txt)
    os.system(f'openssl enc -aes-256-cbc -salt -in {os.path.join(FILE_PATH, "sub.txt")} -out {os.path.join(FILE_PATH, "sub.txt.enc")} -k {os.environ.get("ENCRYPT_KEY")}')
    os.remove(os.path.join(FILE_PATH, 'sub.txt'))
    print(sub_txt)
    print(f"{FILE_PATH}/sub.txt.enc saved successfully")
    send_telegram()
    upload_nodes()
    return sub_txt'''
content = content.replace(old_generate_function, new_generate_function)
with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("扩展分流、Shadowsocks支持和加密存储已成功添加")
EOF
export PROTOCOLS="$PROTOCOLS"
export SS_PASSWORD="$SS_PASSWORD"
export ENCRYPT_KEY="$ENCRYPT_KEY"
python3 extended_patch.py
rm extended_patch.py
echo -e "${GREEN}扩展配置已集成${NC}"

# 清理进程
pkill -f "python3 app.py" > /dev/null 2>&1
pkill -f "keep_alive_task.sh" > /dev/null 2>&1
sleep 2

# 启动服务
nohup python3 app.py > app.log.enc 2>&1 &
APP_PID=$!
sleep 2
APP_PID=$(pgrep -f "python3 app.py" | head -1)
if [ -z "$APP_PID" ]; then
    echo -e "${RED}服务启动失败，请检查日志${NC}"
    echo -e "${YELLOW}解密日志: openssl enc -aes-256-cbc -d -in app.log.enc -out app.log -k $ENCRYPT_KEY && tail -f app.log${NC}"
    exit 1
fi
encrypt_file app.log app.log.enc
rm app.log
echo -e "${GREEN}服务已在后台启动，PID: $APP_PID${NC}"
echo -e "${YELLOW}加密日志文件: $(pwd)/app.log.enc${NC}"

# 保活任务（自适应间隔）
if [ "$KEEP_ALIVE_HF" = "true" ]; then
    echo -e "${BLUE}正在创建并启动 Hugging Face API 保活任务...${NC}"
    echo "#!/bin/bash" > keep_alive_task.sh
    echo "export HF_TOKEN=\"$HF_TOKEN\"" >> keep_alive_task.sh
    echo "while true; do" >> keep_alive_task.sh
    echo "  API_PATH=\"https://huggingface.co/api/${HF_REPO_TYPE}/${HF_REPO_ID}\"" >> keep_alive_task.sh
    echo "  start_time=\$(date +%s)" >> keep_alive_task.sh
    echo "  status_code=\$(curl -s -o /dev/null -w \"%{http_code}\" --header \"Authorization: Bearer \$HF_TOKEN\" \"\$API_PATH\")" >> keep_alive_task.sh
    echo "  end_time=\$(date +%s)" >> keep_alive_task.sh
    echo "  response_time=\$((end_time - start_time))" >> keep_alive_task.sh
    echo "  if [ \"\$status_code\" -eq 200 ]; then" >> keep_alive_task.sh
    echo "    echo \"Hugging Face API 保活成功 (仓库: $HF_REPO_ID, 类型: $HF_REPO_TYPE, 状态码: 200) - \$(date '+%Y-%m-%d %H:%M:%S')\" > keep_alive_status.log" >> keep_alive_task.sh
    echo "    interval=\$((response_time < 2 ? 30 : response_time < 5 ? 60 : 300))" >> keep_alive_task.sh
    echo "  else" >> keep_alive_task.sh
    echo "    echo \"Hugging Face API 保活失败 (仓库: $HF_REPO_ID, 类型: $HF_REPO_TYPE, 状态码: \$status_code) - \$(date '+%Y-%m-%d %H:%M:%S')\" > keep_alive_status.log" >> keep_alive_task.sh
    echo "    interval=30" >> keep_alive_task.sh
    echo "  fi" >> keep_alive_task.sh
    echo "  openssl enc -aes-256-cbc -salt -in keep_alive_status.log -out keep_alive_status.log.enc -k \"$ENCRYPT_KEY\"" >> keep_alive_task.sh
    echo "  rm keep_alive_status.log" >> keep_alive_task.sh
    echo "  sleep \$interval" >> keep_alive_task.sh
    echo "done" >> keep_alive_task.sh
    chmod +x keep_alive_task.sh
    nohup ./keep_alive_task.sh >/dev/null 2>&1 &
    KEEPALIVE_PID=$!
    echo -e "${GREEN}Hugging Face API 保活任务已启动 (PID: $KEEPALIVE_PID)。${NC}"
fi

# 服务监控
echo -e "${BLUE}设置服务监控...${NC}"
echo "*/5 * * * * root pgrep -f 'python3 app.py' || (cd $(pwd) && nohup python3 app.py > app.log.enc 2>&1 &)" | sudo tee /etc/cron.d/xray_monitor > /dev/null

echo -e "${BLUE}等待节点信息生成...${NC}"
MAX_WAIT=300
WAIT_COUNT=0
NODE_INFO=""
while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    if [ -f ".cache/sub.txt.enc" ]; then
        decrypt_file ".cache/sub.txt.enc" /tmp/sub.txt
        NODE_INFO=$(cat /tmp/sub.txt 2>/dev/null)
        rm /tmp/sub.txt
    elif [ -f "sub.txt.enc" ]; then
        decrypt_file "sub.txt.enc" /tmp/sub.txt
        NODE_INFO=$(cat /tmp/sub.txt 2>/dev/null)
        rm /tmp/sub.txt
    fi
    if [ -n "$NODE_INFO" ]; then
        echo -e "${GREEN}节点信息已生成！${NC}"
        break
    fi
    curl -s https://speed.cloudflare.com/__down?bytes=1 > /dev/null && break
    if [ $((WAIT_COUNT % 30)) -eq 0 ]; then
        MINUTES=$((WAIT_COUNT / 60))
        SECONDS=$((WAIT_COUNT % 60))
        echo -e "${YELLOW}已等待 ${MINUTES}分${SECONDS}秒，继续等待...${NC}"
    fi
    sleep 5
    WAIT_COUNT=$((WAIT_COUNT + 5))
done

if [ -z "$NODE_INFO" ]; then
    echo -e "${RED}等待超时！尝试重试Argo隧道...${NC}"
    for i in {1..3}; do
        pkill -f "python3 app.py"
        nohup python3 app.py > app.log.enc 2>&1 &
        sleep 30
        if [ -f ".cache/sub.txt.enc" ]; then
            decrypt_file ".cache/sub.txt.enc" /tmp/sub.txt
            NODE_INFO=$(cat /tmp/sub.txt 2>/dev/null)
            rm /tmp/sub.txt
            break
        fi
    done
    if [ -z "$NODE_INFO" ]; then
        echo -e "${RED}节点信息生成失败，请检查日志${NC}"
        echo -e "${YELLOW}解密日志: openssl enc -aes-256-cbc -d -in app.log.enc -out app.log -k $ENCRYPT_KEY && tail -f app.log${NC}"
        exit 1
    fi
fi

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
echo -e "服务端口: ${BLUE}$(grep "PORT = int" app.py | grep -o "or [0-9]*" | cut -d" " -f2)${NC}"
echo -e "UUID: ${BLUE}$UUID_INPUT${NC}"
echo -e "订阅路径: ${BLUE}/$(grep "SUB_PATH = " app.py | cut -d"'" -f4)${NC}"

PUBLIC_IP=$(curl -s https://api.ipify.org 2>/dev/null || echo "获取失败")
if [ "$PUBLIC_IP" != "获取失败" ]; then
    echo -e "订阅地址: ${GREEN}http://$PUBLIC_IP:$(grep "PORT = int" app.py | grep -o "or [0-9]*" | cut -d" " -f2)/$(grep "SUB_PATH = " app.py | cut -d"'" -f4)${NC}"
    echo -e "管理面板: ${GREEN}http://$PUBLIC_IP:$(grep "PORT = int" app.py | grep -o "or [0-9]*" | cut -d" " -f2)${NC}"
fi
echo -e "本地订阅: ${GREEN}http://localhost:$(grep "PORT = int" app.py | grep -o "or [0-9]*" | cut -d" " -f2)/$(grep "SUB_PATH = " app.py | cut -d"'" -f4)${NC}"
echo -e "本地面板: ${GREEN}http://localhost:$(grep "PORT = int" app.py | grep -o "or [0-9]*" | cut -d" " -f2)${NC}"

echo -e "${YELLOW}=== 节点信息 ===${NC}"
DECODED_NODES=$(echo "$NODE_INFO" | base64 -d 2>/dev/null || echo "$NODE_INFO")
echo -e "${GREEN}节点配置:${NC}"
echo "$DECODED_NODES"
echo -e "${GREEN}订阅链接:${NC}"
echo "$NODE_INFO"
echo -e "${YELLOW}订阅文件已加密，查看需解密: openssl enc -aes-256-cbc -d -in sub.txt.enc -out sub.txt -k [your_password]${NC}"

SAVE_INFO="========================================
                      节点信息保存
========================================
部署时间: $(date)
UUID: $UUID_INPUT
服务端口: $(grep "PORT = int" app.py | grep -o "or [0-9]*" | cut -d" " -f2)
订阅路径: /$(grep "SUB_PATH = " app.py | cut -d"'" -f4)
启用协议: $PROTOCOLS
=== 访问地址 ===
订阅地址: http://$PUBLIC_IP:$(grep "PORT = int" app.py | grep -o "or [0-9]*" | cut -d" " -f2)/$(grep "SUB_PATH = " app.py | cut -d"'" -f4)
管理面板: http://$PUBLIC_IP:$(grep "PORT = int" app.py | grep -o "or [0-9]*" | cut -d" " -f2)
本地订阅: http://localhost:$(grep "PORT = int" app.py | grep -o "or [0-9]*" | cut -d" " -f2)/$(grep "SUB_PATH = " app.py | cut -d"'" -f4)
本地面板: http://localhost:$(grep "PORT = int" app.py | grep -o "or [0-9]*" | cut -d" " -f2)
=== 节点信息 ===
$DECODED_NODES
=== 订阅链接 ===
$NODE_INFO
=== 管理命令 ===
解密日志: openssl enc -aes-256-cbc -d -in app.log.enc -out app.log -k [your_password]
查看日志: tail -f app.log
停止主服务: pkill -f \"python3 app.py\"
重启主服务: pkill -f \"python3 app.py\" && nohup python3 app.py > app.log.enc 2>&1 &
查看进程: ps aux | grep app.py
停止保活服务: pkill -f keep_alive_task.sh && rm keep_alive_task.sh keep_alive_status.log.enc
=== 分流说明 ===
- 支持YouTube、Netflix、Twitch、Bilibili、Douyin、社交平台、游戏平台
- 动态分流规则: 编辑 rules.json"
echo "$SAVE_INFO" > /tmp/node_info
encrypt_file /tmp/node_info "$NODE_INFO_FILE"
rm /tmp/node_info
echo -e "${GREEN}节点信息已加密保存到 $NODE_INFO_FILE${NC}"
echo -e "${YELLOW}查看: bash $0 -v${NC}"
echo -e "${GREEN}部署完成！感谢使用！${NC}"
exit 0
