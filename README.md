## HiFi-GAN Neural Vocoder Model

Fourth assignment on DLA (Deep Learning in Audio) HSE course.

### Reproduce model

1. Clone repository:
```bash
git clone https://github.com/silentz/vocoder.git
```

2. Cd into repository root:
```bash
cd vocoder
```

3. Create and activate virtualenv:
```bash
virtualenv --python=python3 venv
source venv/bin/activate
```

4. Install required packages:
```bash
pip install -r requirements.txt
```

5. Train model:
```bash
./train.sh
```

6. Test model:
```bash
./test.sh
```
