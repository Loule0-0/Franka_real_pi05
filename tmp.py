from modelscope import snapshot_download

model_dir = snapshot_download(
    'hairuoliu/pi05_base',
    local_dir='/home/server/Desktop/vla/model/pi05_base'
)

print(model_dir)