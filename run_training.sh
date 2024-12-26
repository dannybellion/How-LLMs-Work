#!/bin/bash
cd shakespeare
cat > config.json << 'EOF'
{"batch_size": 32, "block_size": 16, "max_iters": 10000, "eval_interval": 500, "learning_rate": 0.0001, "n_embed": 64, "n_heads": 4, "n_layer": 1, "dropout": 0.2}
EOF
python cloud_train.py config.json
