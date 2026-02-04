FROM node:20

ARG TZ
ENV TZ="$TZ"

ARG CLAUDE_CODE_VERSION=latest

# Install basic development tools, iptables/ipset, and ngspice
RUN apt-get update && apt-get install -y --no-install-recommends \
  less \
  git \
  procps \
  sudo \
  fzf \
  zsh \
  man-db \
  unzip \
  gnupg2 \
  gh \
  iptables \
  ipset \
  iproute2 \
  dnsutils \
  aggregate \
  jq \
  nano \
  vim \
  curl \
  wget \
  ca-certificates \
  python3 \
  ngspice \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

# Ensure default node user has access to /usr/local/share
RUN mkdir -p /usr/local/share/npm-global && \
  chown -R node:node /usr/local/share

ARG USERNAME=node

# Persist bash history
RUN SNIPPET="export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" \
  && mkdir /commandhistory \
  && touch /commandhistory/.bash_history \
  && chown -R $USERNAME /commandhistory

# Set `DEVCONTAINER` environment variable to help with orientation
ENV DEVCONTAINER=true

# Create workspace and config directories and set permissions
RUN mkdir -p /workspace /home/node/.claude && \
  chown -R node:node /workspace /home/node/.claude

WORKDIR /workspace

ARG GIT_DELTA_VERSION=0.18.2
RUN ARCH=$(dpkg --print-architecture) && \
  wget "https://github.com/dandavison/delta/releases/download/${GIT_DELTA_VERSION}/git-delta_${GIT_DELTA_VERSION}_${ARCH}.deb" && \
  sudo dpkg -i "git-delta_${GIT_DELTA_VERSION}_${ARCH}.deb" && \
  rm "git-delta_${GIT_DELTA_VERSION}_${ARCH}.deb"

# ── Julia via juliaup ──────────────────────────────────────────────────────────
ARG JULIA_VERSION=1.11
ENV JULIAUP_DEPOT_PATH="/opt/juliaup"
ENV JULIA_DEPOT_PATH="/opt/julia-depot"

RUN mkdir -p "$JULIAUP_DEPOT_PATH" "$JULIA_DEPOT_PATH" && \
  curl -fsSL https://install.julialang.org -o /tmp/julia-installer.sh && \
  bash /tmp/julia-installer.sh -y --default-channel "$JULIA_VERSION" && \
  rm /tmp/julia-installer.sh && \
  # Symlink juliaup and julia into PATH for all users
  ln -sf /root/.juliaup/bin/juliaup /usr/local/bin/juliaup && \
  ln -sf /root/.juliaup/bin/julia   /usr/local/bin/julia && \
  # Make the installation accessible to non-root users
  chmod -R a+rX /root/.juliaup && \
  chmod -R a+rwX "$JULIA_DEPOT_PATH"

# ── OpenVAF Reloaded (prebuilt binary) ────────────────────────────────────────
ARG OPENVAF_VERSION=osdi_0.4-146-g75df7b36
RUN wget -q "https://fides.fe.uni-lj.si/openvaf/download/openvaf-reloaded-${OPENVAF_VERSION}-linux_x64.tar.gz" \
      -O /tmp/openvaf.tar.gz && \
  mkdir -p /opt/openvaf && \
  tar xzf /tmp/openvaf.tar.gz -C /opt/openvaf && \
  ln -sf /opt/openvaf/openvaf-r /usr/local/bin/openvaf-r && \
  rm /tmp/openvaf.tar.gz

# ── VACASK (prebuilt .deb package) ────────────────────────────────────────────
ARG VACASK_VERSION=0.3.2
ARG VACASK_TAG=_0.3.2
RUN wget -q "https://codeberg.org/arpadbuermen/VACASK/releases/download/${VACASK_TAG}/vacask_${VACASK_VERSION}_amd64.deb" \
      -O /tmp/vacask.deb && \
  apt-get update && \
  apt-get install -y --no-install-recommends /tmp/vacask.deb && \
  apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/vacask.deb

# ── Non-root user setup ───────────────────────────────────────────────────────
USER node

# Install global packages
ENV NPM_CONFIG_PREFIX=/usr/local/share/npm-global
ENV PATH=$PATH:/usr/local/share/npm-global/bin

# Set the default shell to zsh
ENV SHELL=/bin/zsh

# Set the default editor
ENV EDITOR=nano
ENV VISUAL=nano

# Default powerline10k theme
ARG ZSH_IN_DOCKER_VERSION=1.2.0
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v${ZSH_IN_DOCKER_VERSION}/zsh-in-docker.sh)" -- \
  -p git \
  -p fzf \
  -a "source /usr/share/doc/fzf/examples/key-bindings.zsh" \
  -a "source /usr/share/doc/fzf/examples/completion.zsh" \
  -a "export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" \
  -x

# Install Claude Code
RUN npm install -g @anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}

# Copy and set up firewall script
COPY init-firewall.sh /usr/local/bin/
USER root
RUN chmod +x /usr/local/bin/init-firewall.sh && \
  echo "node ALL=(root) NOPASSWD: /usr/local/bin/init-firewall.sh" > /etc/sudoers.d/node-firewall && \
  chmod 0440 /etc/sudoers.d/node-firewall
USER node
