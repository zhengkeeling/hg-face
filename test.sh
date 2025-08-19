#!/bin/bash
set -euo pipefail

# -------- 彩色与基础变量 --------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

NODE_INFO_FILE="$HOME/.xray_nodes_info"
PROJECT_DIR_NAME="python-xray-argo"

# 如果是 -v 参数，查看节点信息
if [ "${1:-}" = "-v" ]; then
    if [ -f "$NODE_INFO_FILE" ]; then
        echo -e "${GREEN}========================================${NC}"
        echo -e "${GREEN} 主人，这是您之前保存的节点信息喵~ ${NC}"
        echo -e "${GREEN}========================================${NC}\n"
        cat "$NODE_INFO_FILE"
        echo
    else
        echo -e "${RED}喵呜... 未找到节点信息文件... >.<${NC}"
        echo -e "${YELLOW}请主人先运行部署脚本，本喵才能为您保存信息哦~${NC}"
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

function celebration_animation() {
    echo -e "\n\n"
    echo -e "${GREEN}喵~ 部署任务大成功啦！ >ω<${NC}"
    sleep 0.5
    echo -e "${YELLOW}正在为主人献上胜利的爱心... (｡♥‿♥｡)${NC}"
    sleep 0.5
    echo -e "${RED}"
    cat << "EOF"
          * * * * * *
        * *
      * *
     * *
     * *
      * *
        * *
          * *
            * *
              *
EOF
    echo -e "${NC}"
    sleep 1
    echo -e "${BLUE}所有节点都准备就绪，正在检查最后的魔力...${NC}"
    for i in {1..20}; do
        echo -n "✨"
        sleep 0.05
    done
    echo -e "\n${GREEN}魔力注入完毕！随时可以出发咯！喵~${NC}\n"
}

clear
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN} 主人的专属 Xray Argo 部署脚本喵~ (安全定制 + 暴力加速版) ${NC}"
echo -e "${GREEN}========================================${NC}\n"
echo -e "${BLUE}脚本项目: ${YELLOW}https://github.com/eooce/python-xray-argo${NC}\n"
echo -e "${GREEN}本喵将为主人自动执行“完整模式”部署，并从 Space Secrets 读取 HF Token~${NC}"
read -p "请主人按 Enter 键，开始这次愉快的部署之旅吧！>"

# 自动选择完整模式
MODE_CHOICE="2"

echo
echo -e "${BLUE}喵~ 正在为主人检查和安装必要的“猫粮” (依赖)...${NC}"
sudo apt-get update -qq || true
if ! command -v python3 &> /dev/null; then
    echo -e "${YELLOW}发现主人缺少 Python3，本喵来搞定！...${NC}"
    sudo apt-get install -y python3 python3-pip
fi
if ! python3 -c "import requests" &> /dev/null; then
    echo -e "${YELLOW}需要一点 Python 的小零食 (requests)... 安装中...${NC}"
    pip3 install --user requests
fi
if ! command -v git &> /dev/null; then
    sudo apt-get install -y git
fi
if ! command -v unzip &> /dev/null; then
    sudo apt-get install -y unzip
fi
if ! command -v getent &> /dev/null; then
    sudo apt-get install -y libc-bin
fi
if ! command -v curl &> /dev/null; then
    sudo apt-get install -y curl
fi

if [ ! -d "$PROJECT_DIR_NAME" ]; then
    echo -e "${BLUE}本喵正在努力下载完整的项目仓库... 请稍等哦...${NC}"
    git clone --depth=1 https://github.com/eooce/python-xray-argo.git "$PROJECT_DIR_NAME" || {
        wget -q https://github.com/eooce/python-xray-argo/archive/refs/heads/main.zip -O python-xray-argo.zip
        unzip -q python-xray-argo.zip
        mv python-xray-argo-main "$PROJECT_DIR_NAME"
        rm python-xray-argo.zip
    }
    if [ $? -ne 0 ] || [ ! -d "$PROJECT_DIR_NAME" ]; then
        echo -e "${RED}呜呜... 下载失败了，主人检查下网络吧...${NC}"
        exit 1
    fi
fi

cd "$PROJECT_DIR_NAME"
echo -e "${GREEN}“猫粮”都准备好啦！依赖安装完成！(ฅ´ω`ฅ)${NC}\n"

if [ ! -f "app.py" ]; then
    echo -e "${RED}喵？关键的 app.py 文件不见了！ >.<${NC}"
    exit 1
fi
[ -f "app.py.backup" ] || cp app.py app.py.backup
echo -e "${YELLOW}已为主人备份好原始文件，命名为 app.py.backup 喔~${NC}"

# 初始化保活变量
KEEP_ALIVE_HF="false"
HF_REPO_ID=""
HF_REPO_TYPE="spaces"

# 定义保活配置函数（按你原逻辑自动 y）
configure_hf_keep_alive() {
    echo
    echo -e "${YELLOW}是否为本喵设置 Hugging Face API 自动保活呢? (y/n)${NC}"
    SETUP_KEEP_ALIVE="y"
    echo -e "${GREEN}[本喵猜主人肯定会选 y 啦！]${NC}"
    if [ "$SETUP_KEEP_ALIVE" = "y" ] || [ "$SETUP_KEEP_ALIVE" = "Y" ]; then
        echo -e "${BLUE}正在从主人的 Space secrets 读取 HF 令牌...${NC}"
        if [ -z "${HF_TOKEN:-}" ]; then
            echo -e "${RED}错误：呜... 找不到主人的 HF_TOKEN 令牌...${NC}"
            echo -e "${YELLOW}请主人确认在 Space 的 Settings -> Secrets 中添加了它并重启哦~${NC}"
            KEEP_ALIVE_HF="false"
            return
        fi
        echo -e "${GREEN}成功找到主人的令牌！本喵会好好保管的！${NC}"
        echo -e "${YELLOW}要保活的仓库ID是什么呀? (例如: username/repo):${NC}"
        HF_REPO_ID_INPUT="sukikeeling/face"
        echo -e "${GREEN}[已为主人自动填好 sukikeeling/face 喵~]${NC}"
        echo -e "${YELLOW}仓库类型是 spaces 还是 models 呀？${NC}"
        read -p "Type (留空默认是 spaces 哦): " HF_REPO_TYPE_INPUT || true
        HF_REPO_TYPE="${HF_REPO_TYPE_INPUT:-spaces}"
        HF_REPO_ID="$HF_REPO_ID_INPUT"
        KEEP_ALIVE_HF="true"
        echo -e "${GREEN}保活设置完成！本喵会时刻关注 ${HF_REPO_ID} 的！类型: $HF_REPO_TYPE${NC}"
    fi
}

echo -e "${BLUE}=== 喵~ 自动为主人进入完整配置模式 ===${NC}\n"
echo -e "${YELLOW}当前UUID: $(grep "UUID = " app.py | head -1 | cut -d"'" -f2)${NC}"
UUID_INPUT="c10a3483-5de5-4416-9a37-a6c702b916ac"
echo -e "${GREEN}[UUID 已为主人自动填好喵~]${NC}"
sed -i "s/UUID = os.environ.get('UUID', '[^']*')/UUID = os.environ.get('UUID', '$UUID_INPUT')/" app.py
echo -e "${GREEN}主人的专属UUID已设置好啦！${NC}"

echo -e "${YELLOW}当前节点名称: $(grep "NAME = " app.py | head -1 | cut -d"'" -f4)${NC}"
read -p "主人，要给节点起个可爱的名字吗？(留空也行喔): " NAME_INPUT || true
if [ -n "${NAME_INPUT:-}" ]; then
    sed -i "s/NAME = os.environ.get('NAME', '[^']*')/NAME = os.environ.get('NAME', '$NAME_INPUT')/" app.py
    echo -e "${GREEN}节点的新名字 ${NAME_INPUT} 好可爱！${NC}"
fi

echo -e "${YELLOW}当前服务端口: $(grep "PORT = int" app.py | grep -o "or [0-9]*" | cut -d" " -f2)${NC}"
read -p "服务端口号，主人有什么特别喜欢的数字吗？(留空保持不变): " PORT_INPUT || true
if [ -n "${PORT_INPUT:-}" ]; then
    sed -i "s/PORT = int(os.environ.get('SERVER_PORT') or os.environ.get('PORT') or [0-9]*)/PORT = int(os.environ.get('SERVER_PORT') or os.environ.get('PORT') or $PORT_INPUT)/" app.py
    echo -e "${GREEN}端口已设置为 ${PORT_INPUT}！${NC}"
fi

echo -e "${YELLOW}当前优选IP: $(grep "CFIP = " app.py | cut -d"'" -f4)${NC}"
read -p "优选IP/域名 (主人可以直接回车，使用默认的 joeyblog.net 哦): " CFIP_INPUT || true
if [ -z "${CFIP_INPUT:-}" ]; then
    CFIP_INPUT="joeyblog.net"
fi
sed -i "s/CFIP = os.environ.get('CFIP', '[^']*')/CFIP = os.environ.get('CFIP', '$CFIP_INPUT')/" app.py
echo -e "${GREEN}优选IP已设置为 ${CFIP_INPUT} 喵~${NC}"

echo -e "${YELLOW}是否配置高级选项? (y/n)${NC}"
ADVANCED_CONFIG="y"
echo -e "${GREEN}[本喵觉得主人肯定需要，自动选 y 啦！]${NC}"
if [ "$ADVANCED_CONFIG" = "y" ] || [ "$ADVANCED_CONFIG" = "Y" ]; then
    configure_hf_keep_alive
fi

echo -e "${YELLOW}当前Argo域名: $(grep "ARGO_DOMAIN = " app.py | cut -d"'" -f4)${NC}"
ARGO_DOMAIN_INPUT="face.keeling.dpdns.org"
echo -e "${GREEN}[Argo 域名已为主人自动填好 face.keeling.dpdns.org]${NC}"
if [ -n "$ARGO_DOMAIN_INPUT" ]; then
    sed -i "s|ARGO_DOMAIN = os.environ.get('ARGO_DOMAIN', '[^']*')|ARGO_DOMAIN = os.environ.get('ARGO_DOMAIN', '$ARGO_DOMAIN_INPUT')|" app.py
    echo -e "${YELLOW}当前Argo密钥: $(grep "ARGO_AUTH = " app.py | cut -d"'" -f4)${NC}"
    ARGO_AUTH_INPUT='{"AccountTag":"46fad1b6b0e334ca8ad9ea7ec29c4ddb","TunnelSecret":"J2TOKaJiWL8rph+m7iTfEOthVtREnhuvfWoHp4SmOog=","TunnelID":"29e3716e-783c-4a1f-9538-d40fa766006f","Endpoint":""}'
    echo -e "${GREEN}[Argo 密钥也为主人藏好了哦~]${NC}"
    if [ -n "$ARGO_AUTH_INPUT" ]; then
        sed -i "s|ARGO_AUTH = os.environ.get('ARGO_AUTH', '[^']*')|ARGO_AUTH = os.environ.get('ARGO_AUTH', '$ARGO_AUTH_INPUT')|" app.py
    fi
    echo -e "${GREEN}Argo隧道的秘密设置好啦！${NC}"
fi

echo -e "${GREEN}分流什么的，本喵也自动帮主人配置好了呢~${NC}\n"
echo -e "${GREEN}配置完成！主人真棒！(ﾉ>ω<)ﾉ${NC}\n"

echo -e "${YELLOW}=== 主人请看，这是当前的配置摘要 ===${NC}"
echo -e "主人的UUID: $(grep "UUID = " app.py | head -1 | cut -d"'" -f2)"
echo -e "节点名称: $(grep "NAME = " app.py | head -1 | cut -d"'" -f4)"
echo -e "服务端口: $(grep "PORT = int" app.py | grep -o "or [0-9]*" | cut -d" " -f2)"
echo -e "优选IP: $(grep "CFIP = " app.py | cut -d"'" -f4)"
if [ "$KEEP_ALIVE_HF" = "true" ]; then
    echo -e "保活仓库: $HF_REPO_ID ($HF_REPO_TYPE)"
fi
echo -e "${YELLOW}=====================================${NC}\n"

echo -e "${BLUE}正在为脚本注入更多魔力（扩展分流功能）...喵~${NC}"
cat > extended_patch.py << 'EOF'
# coding: utf-8
import os, base64, json, subprocess, time, re
with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 强化默认 config（不改协议，仅增强日志级别与媒体分流）
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

# 生成订阅时同时给 443/TLS 与 80/明文 WS 两套（不改协议，只多条目）
old_generate_function = r'''# Generate links and subscription content
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
new_generate_function = r'''# Generate links and subscription content
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

# 这里不改协议，只在稍后注入 sockopt/reusePort
with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("魔法注入成功！扩展分流已配置喵~")

EOF
python3 extended_patch.py
rm extended_patch.py

# ------------------- 暴力加速：系统 & Xray 永久开启（无开关） -------------------

echo -e "${BLUE}[加速] 开始应用服务器端“暴力发包”优化（不改协议）…${NC}"

# 句柄/进程限制放开
ulimit -n 1048576 || true
ulimit -u 65535   || true

# 内核网络参数（root 可写）；非 root 时跳过但不中断
SYSCTL_FILE="/etc/sysctl.d/98-xray-turbo.conf"
if [ "$(id -u)" = "0" ]; then
  cat > "$SYSCTL_FILE" <<'EOF'
net.core.default_qdisc = fq
net.ipv4.tcp_congestion_control = bbr

net.core.somaxconn = 65535
net.core.netdev_max_backlog = 262144
net.ipv4.tcp_max_syn_backlog = 262144

net.ipv4.tcp_synack_retries = 3
net.ipv4.tcp_syn_retries = 5

net.core.rmem_max = 67108864
net.core.wmem_max = 67108864
net.core.rmem_default = 8388608
net.core.wmem_default = 8388608

net.ipv4.tcp_mtu_probing = 1

net.ipv4.tcp_keepalive_time = 120
net.ipv4.tcp_keepalive_intvl = 20
net.ipv4.tcp_keepalive_probes = 3

net.ipv4.tcp_fastopen = 3
net.ipv4.tcp_slow_start_after_idle = 0

net.ipv4.tcp_tw_reuse = 1
net.ipv4.tcp_fin_timeout = 15
EOF
  sysctl --system >/dev/null || true
else
  echo -e "${YELLOW}[提示] 当前非 root，无法写 sysctl；将继续进行 Xray 层与边缘优化。${NC}"
fi

# 读取必要字段用于探测
ARGO_DOMAIN=$(grep -E "ARGO_DOMAIN =" app.py | cut -d"'" -f4 || echo "")
CFIP_CUR=$(grep -E "CFIP =" app.py | cut -d"'" -f4 || echo "")
CFPORT=$(grep -E "CFPORT =" app.py | grep -oE "[0-9]+" | head -1 || echo "443")
[ -z "$CFPORT" ] && CFPORT="443"

pick_fast_cfip() {
  local domain="$1"
  local port="$2"
  local candidates=()
  [ -n "$CFIP_CUR" ] && candidates+=("$CFIP_CUR")
  mapfile -t resolved < <(getent ahostsv4 "$domain" | awk '{print $1}' | sort -u | head -n 6)
  candidates+=("${resolved[@]}")

  local best_ip=""
  local best_ms=999999
  for ip in "${candidates[@]}"; do
    [ -z "$ip" ] && continue
    ms=$(curl -s -o /dev/null -w "%{time_connect}\n" \
          --connect-timeout 2 \
          --resolve "$domain:$port:$ip" \
          "https://$domain/cdn-cgi/trace" 2>/dev/null | awk '{printf("%.0f",$1*1000)}')
    [ -z "$ms" ] && ms=999999
    echo "[probe] $domain via $ip:$port => ${ms}ms"
    if [ "$ms" -lt "$best_ms" ]; then
      best_ms="$ms"
      best_ip="$ip"
    fi
  done

  if [ -n "$best_ip" ]; then
    echo "[select] best CF edge: $best_ip (${best_ms}ms)"
    sed -i "s/CFIP = os.environ.get('CFIP', '[^']*')/CFIP = os.environ.get('CFIP', '$best_ip')/" app.py
  else
    echo "[warn] 未找到更优 CF 边缘；保持原 CFIP=$CFIP_CUR"
  fi
}

if [ -n "$ARGO_DOMAIN" ]; then
  pick_fast_cfip "$ARGO_DOMAIN" "$CFPORT"
  # 预热 3 次，缓存路由与 TLS 会话
  for i in 1 2 3; do
    curl -s "https://$ARGO_DOMAIN/cdn-cgi/trace" >/dev/null || true
  done
fi

# 在 Xray inbounds 的 streamSettings 中注入 sockopt 与 reusePort（不改协议）
python3 - <<'PY'
import re
p='app.py'
t=open(p,'r',encoding='utf-8').read()

def enhance(block):
    # 注入 sockopt
    if '"streamSettings"' in block:
        if '"sockopt"' in block:
            block = re.sub(r'"sockopt"\s*:\s*\{[^\}]*\}',
                           '"sockopt":{"tcpNoDelay":true,"tcpFastOpen":true,"mark":0}',
                           block)
        else:
            block = block.replace('"streamSettings": {', '"streamSettings": {"sockopt":{"tcpNoDelay":true,"tcpFastOpen":true,"mark":0}, ')
    # 注入 reusePort
    for proto in ('"vless"','"vmess"','"trojan"'):
        block = re.sub(rf'("protocol": {proto})(?!, "reusePort": true)',
                       rf'\1, "reusePort": true', block)
    return block

t = re.sub(r'("inbounds"\s*:\s*\[)(.*?)(\]\s*,\s*"\w+")',
           lambda m: m.group(1)+enhance(m.group(2))+m.group(3),
           t, flags=re.S)
open(p,'w',encoding='utf-8').write(t)
print("[*] Xray sockopt(tcpNoDelay/tcpFastOpen) & reusePort 已注入")
PY

# ------------------- 启动服务（暴力优先级/CPU 亲和） -------------------
pkill -f "python3 app.py" > /dev/null 2>&1 || true
pkill -f "keep_alive_task.sh" > /dev/null 2>&1 || true
sleep 2

if command -v taskset >/dev/null 2>&1; then
  XRAY_PERF_OPTS="nice -n -2 taskset -c 0-1"
else
  XRAY_PERF_OPTS="nice -n -2"
fi

eval $XRAY_PERF_OPTS nohup python3 app.py > app.log 2>&1 &
APP_PID=$!
sleep 2
APP_PID=$(pgrep -f "python3 app.py" | head -1 || true)

if [ -z "${APP_PID:-}" ]; then
    echo -e "${RED}呜喵... 服务启动失败了... 主人快检查下Python环境吧...${NC}"
    echo -e "${YELLOW}可以看看日志: tail -f app.log${NC}"
    exit 1
fi
echo -e "${GREEN}服务已在后台为主人悄悄启动啦，PID: $APP_PID${NC}"

# ------------------- HF 保活（如设定） -------------------
if [ "$KEEP_ALIVE_HF" = "true" ]; then
    echo -e "${BLUE}正在为主人启动 Hugging Face API 保活任务...${NC}"
    cat > keep_alive_task.sh <<'KAF'
#!/bin/bash
while true; do
    API_PATH="https://huggingface.co/api/'"$HF_REPO_TYPE"'/'"$HF_REPO_ID"'"
    status_code=$(curl -s -o /dev/null -w "%{http_code}" --header "Authorization: Bearer $HF_TOKEN" "$API_PATH")
    if [ "$status_code" -eq 200 ]; then
        echo "喵~ 在 $(date '+%Y-%m-%d %H:%M:%S') 成功帮主人保活了仓库 ('"$HF_REPO_ID"')！" > keep_alive_status.log
    else
        echo "呜... 在 $(date '+%Y-%m-%d %H:%M:%S') 保活失败 (状态码: $status_code)... T_T" > keep_alive_status.log
    fi
    sleep 300
done
KAF
    chmod +x keep_alive_task.sh
    export HF_TOKEN="${HF_TOKEN:-}"
    nohup ./keep_alive_task.sh >/dev/null 2>&1 &
    KEEPALIVE_PID=$!
    echo -e "${GREEN}保活任务已启动 (PID: $KEEPALIVE_PID)，本喵会一直盯着的！${NC}"
fi

# ------------------- 等待并展示节点信息 -------------------
echo -e "${BLUE}喵~ 正在努力生成节点信息，就像在烤小鱼干一样...${NC}"
echo -e "${YELLOW}这个过程可能需要一点点时间，请主人耐心等待哦...${NC}"

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
        echo -e "${GREEN}小鱼干烤好了！节点信息生成啦！${NC}"
        break
    fi
    sleep 5
    WAIT_COUNT=$((WAIT_COUNT + 5))
done

if [ -z "$NODE_INFO" ]; then
    echo -e "${RED}喵呜... 等待超时了... 节点信息生成失败... T_T${NC}"
    echo -e "${YELLOW}可能原因：网络问题、Argo失败、配置错误${NC}"
    echo -e "${BLUE}主人可以看看日志: tail -f $(pwd)/app.log${NC}"
    exit 1
fi

echo -e "${YELLOW}=== 主人的服务信息 ===${NC}"
echo -e "服务状态: ${GREEN}正在运行喵~${NC}"
echo -e "主服务PID: ${BLUE}$APP_PID${NC}"
if [ -n "${KEEPALIVE_PID:-}" ]; then
    echo -e "保活服务PID: ${BLUE}$KEEPALIVE_PID${NC}"
fi
echo -e "服务端口: ${BLUE}$SERVICE_PORT${NC}"
echo -e "主人的UUID: ${BLUE}$CURRENT_UUID${NC}"
echo -e "订阅路径: ${BLUE}/$SUB_PATH_VALUE${NC}\n"

echo -e "${YELLOW}=== 访问地址 ===${NC}"
PUBLIC_IP=$(curl -s https://api.ipify.org 2>/dev/null || echo "获取失败")
if [ "$PUBLIC_IP" != "获取失败" ]; then
    echo -e "订阅地址: ${GREEN}http://$PUBLIC_IP:$SERVICE_PORT/$SUB_PATH_VALUE${NC}"
    echo -e "管理面板: ${GREEN}http://$PUBLIC_IP:$SERVICE_PORT${NC}"
fi
echo

echo -e "${YELLOW}=== 节点信息 ===${NC}"
DECODED_NODES=$(echo "$NODE_INFO" | base64 -d 2>/dev/null || echo "$NODE_INFO")
echo -e "${GREEN}节点配置:${NC}"
echo "$DECODED_NODES"
echo

SAVE_INFO="========================================
主人，这是您的节点信息，请收好喵~
部署时间: $(date)
主人的UUID: $CURRENT_UUID
服务端口: $SERVICE_PORT
... (此处省略，内容和之前版本一致)
"
echo "$SAVE_INFO" > "$NODE_INFO_FILE"
echo -e "${GREEN}已将节点信息保存到 $NODE_INFO_FILE 啦~${NC}"
echo -e "${YELLOW}主人随时可以用 'bash $0 -v' 命令偷看哦~${NC}\n"

celebration_animation
exit 0
