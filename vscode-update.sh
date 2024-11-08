# (1). 打开你设备上的命令行；
# (2). 执行以下命令：
# 	code --version
#      输出结果一共有3行，其形式如下所示（例）：
#	x.xx.0			（vscode版本号）
# 	xxxxxxxx		（vscode commit id）
# 	xxx			（设备平台使用的指令集）
# (3). 复制第二行的commit id；
# (4). 使用命令行远程连接到服务器上，把本脚本传上去，然后在自己的目录下运行以下命令：
# 	sudo chmod +x vscode-update.sh && ./vscode-update.sh

cd ~

read -p "请输入你vscode的commit id : " commit

# 新方法

if [ ! -d "~/Hwj/.vscode-server" ] || 
   [ ! -d "~/Hwj/.vscode-server/cli" ] || 
   [ ! -d "~/Hwj/.vscode-server/cli/servers" ] || 
   [ ! -d "~/Hwj/.vscode-server/cli/servers/Stable-$commit" ]; then
	mkdir -p ~/Hwj/.vscode-server/cli/servers/Stable-$commit
fi

cd ~/Hwj/.vscode-server/cli/servers/Stable-$commit
if [ -f "vscode-server-linux-x64.tar.gz" ]; then	
	rm vscode-server-linux-x64.tar.gz
fi
if [ ! -d "server" ]; then
	wget https://vscode.download.prss.microsoft.com/dbazure/download/stable/$commit/vscode-server-linux-x64.tar.gz
	tar -zxf vscode-server-linux-x64.tar.gz
	mv vscode-server-linux-x64 server
	rm vscode-server-linux-x64.tar.gz

fi
cd ~/Hwj/.vscode-server
if [ ! -f "code-$commit" ]; then
	wget https://vscode.download.prss.microsoft.com/dbazure/download/stable/$commit/vscode_cli_alpine_x64_cli.tar.gz
	mv vscode_cli_alpine_x64_cli.tar.gz code-$commit
fi



echo "vscode-server已配置完毕。"
cd ~