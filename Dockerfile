# Source image
FROM python:3.8-slim

ARG TORCH_VERSION=2.0.1
ARG TORCHVISION_VERSION=0.15.2

# Set the working directory in the container
WORKDIR /app

# Install some basic dependencies
RUN apt-get update
RUN apt-get install -y --no-install-recommends tzdata locales zsh tmux tree htop zip unzip libgomp1

# Install Oh My Zsh and configure plugins
RUN yes | sh -c "$(curl -fsSL https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh)" -y && \
    git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions && \
    git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting && \
    sed -i -E "s/(plugins=\(.+)\)/\1\ zsh-syntax-highlighting\)/" ~/.zshrc && \
    sed -i -E "s/(plugins=\(.+)\)/\1\ zsh-autosuggestions\)/" ~/.zshrc && \
    sed -i -E "s/%c/%~/" ~/.oh-my-zsh/themes/robbyrussell.zsh-theme && \
    chsh -s /bin/zsh

# Instatll requirements
RUN pip install -U torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION}
RUN pip install autogluon==0.8.2
RUN pip install torchmetrics==0.9.3

# Copy the current directory contents into the container at /app
COPY . /app
