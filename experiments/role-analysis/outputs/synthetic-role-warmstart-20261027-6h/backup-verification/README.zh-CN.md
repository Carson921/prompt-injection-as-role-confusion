# 私有备份验收

本轮runpod任务根全部新载荷已回传仓库外私有目录，仅排除已在旧完整备份验证保存的old-features。1054普通文件逐一完整SHA匹配独立源清单、42符号链接目标匹配、无额外/缺失项；复制后源端稳定性零差异，两次sync成功。正文、输入、特征和恢复缓存没有复制到Git。

backup-summary.json为本次通过证明；完整源清单、rsync日志、校验清单和原始字节均在private backup-metadata及task/。旧feature依赖、57原始缓存与本轮canonical离线包均原封保留并核对manifest身份。权重和venv未复制，固定版本和SHA已保留。

备份未删除远端文件，也未暂停/关闭实例。后续local-derived报告采用独立清单，避免哈希循环；不改本次已验证task镜像。
