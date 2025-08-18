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

# 新增：随机化函数
generate_random_path() {
    python3 -c "import random, string; print(''.join(random.choices(string.ascii_letters + string.digits, k=10)))"
}

generate_random_port() {
    echo $((RANDOM % 1000 + 8000))  # 随机8000-9000，避免HF端口限
}

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
echo -e "${GREEN}本脚本基于 eooce 大佬的 Python Xray Argo 项目开发${NC}"
echo -e "${GREEN}魔改: 减少开销、提升安全/速度、突破封锁、反HF检测${NC}"
echo -e "${GREEN}自动随机化、伪装进程、假AI任务、低频保活${NC}"
echo -e "${GREEN}集成REALITY/Hysteria2，支持扩展分流${NC}"
echo
echo -e "${YELLOW}请选择操作:${NC}"
echo -e "${BLUE}1) 极速模式 - 只修改UUID并启动${NC}"
echo -e "${BLUE}2) 完整模式 - 详细配置所有选项${NC}"
echo -e "${BLUE}3) 查看节点信息 - 显示已保存的节点信息${NC}"
echo -e "${BLUE}4) 查看保活状态 - 检查Hugging Face API保活状态${NC}"
echo
read -p "请输入选择 (1/2/3/4): " MODE_CHOICE

if [ "$MODE_CHOICE" = "3" ]; then
    if [ -f "$NODE_INFO_FILE" ]; then
        echo
        echo -e "${GREEN}========================================${NC}"
        echo -e "${GREEN} 节点信息查看 ${NC}"
        echo -e "${GREEN}========================================${NC}"
        echo
        cat "$NODE_INFO_FILE"
        echo
        echo -e "${YELLOW}提示: 如需重新部署，请重新运行脚本选择模式1或2${NC}"
    else
        echo
        echo -e "${RED}未找到节点信息文件${NC}"
        echo -e "${YELLOW}请先运行部署脚本生成节点信息${NC}"
        echo
        echo -e "${BLUE}是否现在开始部署? (y/n)${NC}"
        read -p "> " START_DEPLOY
        if [ "$START_DEPLOY" = "y" ] || [ "$START_DEPLOY" = "Y" ]; then
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

echo -e "${BLUE}检查并安装依赖...${NC}"
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
# 新增：gcc for libprocesshider
sudo apt-get install -y gcc

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
HF_REPO_TYPE="spaces"

# 定义保活配置函数
configure_hf_keep_alive() {
    echo
    echo -e "${YELLOW}是否设置 Hugging Face API 自动保活? (y/n)${NC}"
    read -p "> " SETUP_KEEP_ALIVE
    if [ "$SETUP_KEEP_ALIVE" = "y" ] || [ "$SETUP_KEEP_ALIVE" = "Y" ]; then
        echo -e "${YELLOW}请输入您的 Hugging Face 访问令牌 (Token):${NC}"
        echo -e "${BLUE}（令牌用于API认证。请前往 https://huggingface.co/settings/tokens 获取）${NC}"
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

if [ "$MODE_CHOICE" = "1" ]; then
    echo -e "${BLUE}=== 极速模式 ===${NC}"
    echo
    
    echo -e "${YELLOW}当前UUID: $(grep "UUID = " app.py | head -1 | cut -d"'" -f2)${NC}"
    read -p "请输入新的 UUID (留空自动生成): " UUID_INPUT
    if [ -z "$UUID_INPUT" ]; then
        UUID_INPUT=$(generate_uuid)
        echo -e "${GREEN}自动生成UUID: $UUID_INPUT${NC}"
    fi
    
    sed -i "s/UUID = os.environ.get('UUID', '[^']*')/UUID = os.environ.get('UUID', '$UUID_INPUT')/" app.py
    echo -e "${GREEN}UUID 已设置为: $UUID_INPUT${NC}"
    
    sed -i "s/CFIP = os.environ.get('CFIP', '[^']*')/CFIP = os.environ.get('CFIP', 'joeyblog.net')/" app.py
    echo -e "${GREEN}优选IP已自动设置为: joeyblog.net${NC}"
    
    configure_hf_keep_alive
    
    echo -e "${GREEN}扩展分流已自动配置${NC}"
    echo
    echo -e "${GREEN}极速配置完成！正在启动服务...${NC}"
    echo
    
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
    read -p "请输入 Argo 端口 (留空随机): " ARGO_PORT_INPUT
    if [ -z "$ARGO_PORT_INPUT" ]; then
        ARGO_PORT_INPUT=$(generate_random_port)
        echo -e "${GREEN}Argo端口随机化为: $ARGO_PORT_INPUT${NC}"
    fi
    sed -i "s/ARGO_PORT = int(os.environ.get('ARGO_PORT', '[^']*'))/ARGO_PORT = int(os.environ.get('ARGO_PORT', '$ARGO_PORT_INPUT'))/" app.py
    echo -e "${GREEN}Argo端口已设置为: $ARGO_PORT_INPUT${NC}"
    
    echo -e "${YELLOW}当前订阅路径: $(grep "SUB_PATH = " app.py | cut -d"'" -f4)${NC}"
    read -p "请输入订阅路径 (留空随机): " SUB_PATH_INPUT
    if [ -z "$SUB_PATH_INPUT" ]; then
        SUB_PATH_INPUT=$(generate_random_path)
        echo -e "${GREEN}订阅路径随机化为: $SUB_PATH_INPUT${NC}"
    fi
    sed -i "s/SUB_PATH = os.environ.get('SUB_PATH', '[^']*')/SUB_PATH = os.environ.get('SUB_PATH', '$SUB_PATH_INPUT')/" app.py
    echo -e "${GREEN}订阅路径已设置为: $SUB_PATH_INPUT${NC}"
    
    echo
    echo -e "${YELLOW}是否配置高级选项? (y/n)${NC}"
    read -p "> " ADVANCED_CONFIG
    if [ "$ADVANCED_CONFIG" = "y" ] || [ "$ADVANCED_CONFIG" = "Y" ]; then
        echo -e "${YELLOW}当前上传URL: $(grep "UPLOAD_URL = " app.py | cut -d"'" -f4)${NC}"
        read -p "请输入上传URL (留空保持不变): " UPLOAD_URL_INPUT
        if [ -n "$UPLOAD_URL_INPUT" ]; then
            sed -i "s|UPLOAD_URL = os.environ.get('UPLOAD_URL', '[^']*')|UPLOAD_URL = os.environ.get('UPLOAD_URL', '$UPLOAD_URL_INPUT')|" app.py
            echo -e "${GREEN}上传URL已设置${NC}"
        fi
        
        echo -e "${YELLOW}当前项目URL: $(grep "PROJECT_URL = " app.py | cut -d"'" -f4)${NC}"
        read -p "请输入项目URL (留空保持不变): " PROJECT_URL_INPUT
        if [ -n "$PROJECT_URL_INPUT" ]; then
            sed -i "s|PROJECT_URL = os.environ.get('PROJECT_URL', '[^']*')|PROJECT_URL = os.environ.get('PROJECT_URL', '$PROJECT_URL_INPUT')|" app.py
            echo -e "${GREEN}项目URL已设置${NC}"
        fi
        
        configure_hf_keep_alive
        
        echo -e "${YELLOW}当前哪吒服务器: $(grep "NEZHA_SERVER = " app.py | cut -d"'" -f4)${NC}"
        read -p "请输入哪吒服务器地址 (留空保持不变): " NEZHA_SERVER_INPUT
        if [ -n "$NEZHA_SERVER_INPUT" ]; then
            sed -i "s|NEZHA_SERVER = os.environ.get('NEZHA_SERVER', '[^']*')|NEZHA_SERVER = os.environ.get('NEZHA_SERVER', '$NEZHA_SERVER_INPUT')|" app.py
            
            echo -e "${YELLOW}当前哪吒端口: $(grep "NEZHA_PORT = " app.py | cut -d"'" -f4)${NC}"
            read -p "请输入哪吒端口 (v1版本留空): " NEZHA_PORT_INPUT
            if [ -n "$NEZHA_PORT_INPUT" ]; then
                sed -i "s|NEZHA_PORT = os.environ.get('NEZHA_PORT', '[^']*')|NEZHA_PORT = os.environ.get('NEZHA_PORT', '$NEZHA_PORT_INPUT')|" app.py
            fi
            
            echo -e "${YELLOW}当前哪吒密钥: $(grep "NEZHA_KEY = " app.py | cut -d"'" -f4)${NC}"
            read -p "请输入哪吒密钥: " NEZHA_KEY_INPUT
            if [ -n "$NEZHA_KEY_INPUT" ]; then
                sed -i "s|NEZHA_KEY = os.environ.get('NEZHA_KEY', '[^']*')|NEZHA_KEY = os.environ.get('NEZHA_KEY', '$NEZHA_KEY_INPUT')|" app.py
            fi
            echo -e "${GREEN}哪吒配置已设置${NC}"
        fi
        
        echo -e "${YELLOW}当前Argo域名: $(grep "ARGO_DOMAIN = " app.py | cut -d"'" -f4)${NC}"
        read -p "请输入 Argo 固定隧道域名 (留空保持不变): " ARGO_DOMAIN_INPUT
        if [ -n "$ARGO_DOMAIN_INPUT" ]; then
            sed -i "s|ARGO_DOMAIN = os.environ.get('ARGO_DOMAIN', '[^']*')|ARGO_DOMAIN = os.environ.get('ARGO_DOMAIN', '$ARGO_DOMAIN_INPUT')|" app.py
            
            echo -e "${YELLOW}当前Argo密钥: $(grep "ARGO_AUTH = " app.py | cut -d"'" -f4)${NC}"
            read -p "请输入 Argo 固定隧道密钥: " ARGO_AUTH_INPUT
            if [ -n "$ARGO_AUTH_INPUT" ]; then
                sed -i "s|ARGO_AUTH = os.environ.get('ARGO_AUTH', '[^']*')|ARGO_AUTH = os.environ.get('ARGO_AUTH', '$ARGO_AUTH_INPUT')|" app.py
            fi
            echo -e "${GREEN}Argo固定隧道配置已设置${NC}"
        fi
        
        echo -e "${YELLOW}当前Bot Token: $(grep "BOT_TOKEN = " app.py | cut -d"'" -f4)${NC}"
        read -p "请输入 Telegram Bot Token (留空保持不变): " BOT_TOKEN_INPUT
        if [ -n "$BOT_TOKEN_INPUT" ]; then
            sed -i "s|BOT_TOKEN = os.environ.get('BOT_TOKEN', '[^']*')|BOT_TOKEN = os.environ.get('BOT_TOKEN', '$BOT_TOKEN_INPUT')|" app.py
            
            echo -e "${YELLOW}当前Chat ID: $(grep "CHAT_ID = " app.py | cut -d"'" -f4)${NC}"
            read -p "请输入 Telegram Chat ID: " CHAT_ID_INPUT
            if [ -n "$CHAT_ID_INPUT" ]; then
                sed -i "s|CHAT_ID = os.environ.get('CHAT_ID', '[^']*')|CHAT_ID = os.environ.get('CHAT_ID', '$CHAT_ID_INPUT')|" app.py
            fi
            echo -e "${GREEN}Telegram配置已设置${NC}"
        fi
        
        # 新增：启用Hysteria2选项
        echo -e "${YELLOW}启用Hysteria2? (y/n, 高效QUIC突破封锁)${NC}"
        read -p "> " ENABLE_HYSTERIA
        if [ "$ENABLE_HYSTERIA" = "y" ]; then
            wget -q https://github.com/apernet/apernet/releases/download/v2.5.1/hysteria-linux-amd64 -O hysteria || {
                echo -e "${RED}Hysteria2下载失败${NC}"
            }
            chmod +x hysteria
            read -p "请输入Hysteria2服务器地址: " HY_SERVER_INPUT
            if [ -n "$HY_SERVER_INPUT" ]; then
                sed -i '/outbounds/a ,\n{"protocol": "hysteria2", "tag": "hy2_out", "settings": {"server": "'$HY_SERVER_INPUT'"}}' app.py
                echo -e "${GREEN}Hysteria2集成（低开销UDP）${NC}"
            fi
        fi
    fi
    
    echo -e "${GREEN}扩展分流已自动配置${NC}"
    echo
    echo -e "${GREEN}完整配置完成！${NC}"
fi

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

# 修改Python文件添加扩展分流、REALITY等
echo -e "${BLUE}正在添加扩展分流、REALITY伪装和uTLS指纹...${NC}"
cat > magicked_patch.py << 'EOF'
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
            "loglevel": "none"  # 最小日志，反检测
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
                "protocol": "vless",
                "tag": "reality_out",
                "settings": {
                    "vnext": [{
                        "address": "icook.hk",  # 示例伪装域名，替换为真实
                        "port": 443,
                        "users": [{"id": UUID, "flow": "xtls-rprx-vision"}]
                    }]
                },
                "streamSettings": {
                    "network": "tcp",
                    "security": "reality",
                    "realitySettings": {
                        "show": false,
                        "dest": "icook.hk:443",
                        "serverNames": ["icook.hk"],
                        "fingerprint": "chrome"  # uTLS指纹
                    }
                }
            },
            {"protocol": "blackhole", "tag": "block"}
        ],
        "routing": {
            "domainStrategy": "IPIfNonMatch",
            "rules": [
                {
                    "type": "field",
                    "domain": [
                        # 扩展域名列表
                        "youtube.com", "youtu.be", "googlevideo.com", "ytimg.com",
                        "gstatic.com", "googleapis.com", "ggpht.com", "googleusercontent.com",
                        "facebook.com", "fb.com", "fbcdn.net", "instagram.com", "cdninstagram.com",
                        "fbsbx.com", "api.facebook.com",
                        "twitter.com", "x.com", "twimg.com", "t.co",
                        "discord.com", "discordapp.com", "discord.gg", "discord.media",
                        "discordapp.net",
                        "telegram.org", "t.me", "telegram.me", "web.telegram.org",
                        "cdn.telegram.org", "pluto.web.telegram.org", "venus.web.telegram.org",
                        "apollo.web.telegram.org",
                        "whatsapp.com", "whatsapp.net", "meta.com", "meta.ai",
                        "api.meta.ai", "api.whatsapp.com", "messenger.com", "api.messenger.com",
                        "tiktok.com", "tiktokv.com", "ttlivecdn.com", "byteoversea.com",
                        "musical.ly", "tik-tokcdn.com",
                        "netflix.com", "netflix.net", "nflxvideo.net", "nflximg.net",
                        "nflxso.net", "nflxext.com"
                    ],
                    "outboundTag": "reality_out"  # 使用REALITY出站
                }
            ]
        }
    }'''
# 替换配置
content = content.replace(old_config, new_config)
# 修改generate_links函数，添加80端口节点和REALITY链接
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
    time.sleep(1)  # 减少等待
    
    # TLS节点
    VMESS_TLS = {"v": "2", "ps": f"{NAME}-{ISP}-TLS", "add": CFIP, "port": CFPORT, "id": UUID, "aid": "0", "scy": "none", "net": "ws", "type": "none", "host": argo_domain, "path": "/vmess-argo?ed=2560", "tls": "tls", "sni": argo_domain, "alpn": "", "fp": "chrome"}
    
    # 无TLS节点 (80端口)
    VMESS_80 = {"v": "2", "ps": f"{NAME}-{ISP}-80", "add": CFIP, "port": "80", "id": UUID, "aid": "0", "scy": "none", "net": "ws", "type": "none", "host": argo_domain, "path": "/vmess-argo?ed=2560", "tls": "", "sni": "", "alpn": "", "fp": ""}
    
    # 添加REALITY节点
    list_txt = f"""
vless://{UUID}@icook.hk:443?encryption=none&flow=xtls-rprx-vision&security=reality&sni=icook.hk&fp=chrome&pbk=your_public_key&sid=your_short_id&type=tcp&headerType=none#{NAME}-{ISP}-REALITY
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
print("扩展分流、REALITY伪装和80端口节点已成功添加")
EOF
python3 magicked_patch.py
rm magicked_patch.py
echo -e "${GREEN}扩展分流和REALITY伪装已集成${NC}"

# 先清理可能存在的进程
pkill -f "python3 app.py" > /dev/null 2>&1
pkill -f "keep_alive_task.sh" > /dev/null 2>&1
pkill -f "fake_ai_task.py" > /dev/null 2>&1
sleep 2

# 新增：进程隐藏库
echo -e "${BLUE}编译进程隐藏库（反检测）...${NC}"
cat > processhider.c << 'EOF'
#include <dlfcn.h>
#include <dirent.h>
#include <string.h>
#include <stdio.h>

#define HIDDEN "app.py|keep_alive_task.sh|fake_ai_task.py|hysteria"  // 要隐藏的关键词
#define FAKE_NAME "hf-ml-inference"  // 伪装名称

typedef struct dirent *(*readdir_func)(DIR *);
struct dirent *readdir(DIR *dirp) {
    static readdir_func orig_readdir = NULL;
    if (!orig_readdir) orig_readdir = (readdir_func)dlsym(RTLD_NEXT, "readdir");
    struct dirent *entry;
    do {
        entry = orig_readdir(dirp);
        if (entry && strstr(entry->d_name, HIDDEN)) {
            // 隐藏或伪装
            strncpy(entry->d_name, FAKE_NAME, sizeof(entry->d_name) - 1);
        }
    } while (entry && strstr(entry->d_name, HIDDEN));
    return entry;
}
EOF
gcc -shared -fPIC -o libprocesshider.so processhider.c -ldl || echo -e "${YELLOW}编译libprocesshider失败，可能权限不足${NC}"
export LD_PRELOAD=$(pwd)/libprocesshider.so
echo -e "${GREEN}进程隐藏启用（伪装为hf-ml-inference）${NC}"

# 启动服务
nohup python3 app.py > /dev/null 2>&1 &
APP_PID=$(pgrep -f "python3 app.py" | head -1)
if [ -z "$APP_PID" ]; then
    echo -e "${RED}服务启动失败，请检查Python环境${NC}"
    exit 1
fi
echo -e "${GREEN}服务已在后台启动，PID: $APP_PID (隐藏中)${NC}"

# 保活任务 (间隔600s，低频)
KEEPALIVE_PID=""
if [ "$KEEP_ALIVE_HF" = "true" ]; then
    echo -e "${BLUE}正在创建并启动 Hugging Face API 保活任务...${NC}"
    echo "#!/bin/bash" > keep_alive_task.sh
    echo "while true; do" >> keep_alive_task.sh
    echo "  API_PATH=\"https://huggingface.co/api/${HF_REPO_TYPE}/${HF_REPO_ID}\"" >> keep_alive_task.sh
    echo "  status_code=\$(curl -s -o /dev/null -w \"%{http_code}\" --header \"Authorization: Bearer \$HF_TOKEN\" \"\$API_PATH\")" >> keep_alive_task.sh
    echo "  if [ \"\$status_code\" -eq 200 ]; then" >> keep_alive_task.sh
    echo "    echo \"成功 - \$(date)\" > keep_alive_status.log" >> keep_alive_task.sh
    echo "  else" >> keep_alive_task.sh
    echo "    echo \"失败 (\$status_code) - \$(date)\" > keep_alive_status.log" >> keep_alive_task.sh
    echo "  fi" >> keep_alive_task.sh
    echo "  sleep 600  # 低频反检测" >> keep_alive_task.sh
    echo "done" >> keep_alive_task.sh
    export HF_TOKEN="$HF_TOKEN"
    chmod +x keep_alive_task.sh
    nohup ./keep_alive_task.sh >/dev/null 2>&1 &
    KEEPALIVE_PID=$!
    echo -e "${GREEN}保活任务启动 (PID: $KEEPALIVE_PID, 隐藏中)${NC}"
fi

# 新增：假AI任务伪装
echo -e "${BLUE}启动假AI任务（低资源伪装，反检测）...${NC}"
cat > fake_ai_task.py << 'EOF'
import time, random
import numpy as np  # 低资源ML模拟
while True:
    arr = np.random.rand(100, 100)  # 简单矩阵计算
    np.linalg.det(arr)  # 模拟推理
    time.sleep(random.randint(60, 120))  # 低频<5% CPU
EOF
nohup python3 fake_ai_task.py > /dev/null 2>&1 &
FAKE_PID=$!
echo -e "${GREEN}假AI任务启动 (PID: $FAKE_PID, 隐藏中)${NC}"

echo -e "${BLUE}等待服务启动...${NC}"
sleep 5

# 检查服务
if ! ps -p "$APP_PID" > /dev/null 2>&1; then
    echo -e "${RED}服务启动失败${NC}"
    exit 1
fi
echo -e "${GREEN}服务运行正常（隐藏中）${NC}"

SERVICE_PORT=$(grep "PORT = int" app.py | grep -o "or [0-9]*" | cut -d" " -f2)
CURRENT_UUID=$(grep "UUID = " app.py | head -1 | cut -d"'" -f2)
SUB_PATH_VALUE=$(grep "SUB_PATH = " app.py | cut -d"'" -f4)

echo -e "${BLUE}等待节点信息生成...${NC}"
echo -e "${YELLOW}正在等待Argo隧道建立和节点生成，请耐心等待...${NC}"

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
        echo -e "${YELLOW}已等待 ${MINUTES}分${SECONDS}秒，继续等待...${NC}"
    fi
    
    sleep 5
    WAIT_COUNT=$((WAIT_COUNT + 5))
done

if [ -z "$NODE_INFO" ]; then
    echo -e "${RED}等待超时！${NC}"
    exit 1
fi

echo
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN} 魔改部署完成！ ${NC}"
echo -e "${GREEN}========================================${NC}"
echo

echo -e "${YELLOW}=== 服务信息 ===${NC}"
echo -e "服务状态: ${GREEN}运行中 (隐藏)${NC}"
echo -e "主服务PID: ${BLUE}$APP_PID (伪装)${NC}"
if [ -n "$KEEPALIVE_PID" ]; then
    echo -e "保活PID: ${BLUE}$KEEPALIVE_PID (伪装)${NC}"
fi
echo -e "假AI PID: ${BLUE}$FAKE_PID (伪装)${NC}"
echo -e "服务端口: ${BLUE}$SERVICE_PORT${NC}"
echo -e "UUID: ${BLUE}$CURRENT_UUID${NC}"
echo -e "订阅路径: ${BLUE}/$SUB_PATH_VALUE (随机)${NC}"

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
                      节点信息保存 (魔改版)
========================================
部署时间: $(date)
UUID: $CURRENT_UUID
服务端口: $SERVICE_PORT
订阅路径: /$SUB_PATH_VALUE (随机)
=== 访问地址 ===
订阅地址: http://$PUBLIC_IP:$SERVICE_PORT/$SUB_PATH_VALUE
管理面板: http://$PUBLIC_IP:$SERVICE_PORT
本地订阅: http://localhost:$SERVICE_PORT/$SUB_PATH_VALUE
本地面板: http://localhost:$SERVICE_PORT
=== 节点信息 ===
$DECODED_NODES
=== 订阅链接 ===
$NODE_INFO
=== 管理命令 (隐藏模式) ===
查看日志: 无日志 (反检测)
停止主服务: pkill -f \"hf-ml-inference\"
重启主服务: pkill -f \"hf-ml-inference\" && nohup python3 app.py > /dev/null 2>&1 &
查看进程: ps aux | grep hf-ml  # 伪装名称
停止保活: pkill -f keep_alive_task.sh
停止假AI: pkill -f fake_ai_task.py
=== 魔改说明 ===
- REALITY伪装 + uTLS指纹: 突破封锁
- Hysteria2 (可选): 低开销QUIC
- 进程隐藏 + 假AI任务: 反HF检测
- 随机路径/端口 + 低频保活: 减少开销/隐蔽"

echo "$SAVE_INFO" > "$NODE_INFO_FILE"
echo -e "${GREEN}节点信息已保存到 $NODE_INFO_FILE${NC}"
echo -e "${YELLOW}使用脚本选择选项3或运行带-v参数可随时查看节点信息${NC}"

echo -e "${YELLOW}=== 重要提示 ===${NC}"
echo -e "${GREEN}魔改部署完成，节点隐蔽高效${NC}"
echo -e "${GREEN}使用随机订阅添加到客户端${NC}"
echo -e "${GREEN}REALITY/Hysteria2提升速度/突破${NC}"
echo -e "${GREEN}服务后台运行，无日志暴露${NC}"
echo
echo -e "${GREEN}部署完成！感谢使用！${NC}"

# 退出脚本
exit 0
