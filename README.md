## VisualFastMappingBenchmark
The VisualFastMapping Benchmark is an evaluation dataset designed to assess a model's ability to quickly learn and form new visual concepts from very few examples, by leveraging its existing experience and knowledge.
![image/png](https://cdn-uploads.huggingface.co/production/uploads/67c7bcfdfbf67e415602cff7/W3DrX9pcPN3x5M9XI_O61.png)

## Quick Start
Step 1: Use various large models to predict the answer
```bash
# Test scripts to quickly determine whether the code is running smoothly
sh stage1_demo.sh

# Start predict the answer
sh stage1_template_all.sh
```

Step 2: Filter data by diversity and difficulty
```bash
# Test scripts to quickly determine whether the code is running smoothly
sh stage2_demo.sh

# Generate an image with a mask
sh stage2_imgprocesser_all.sh

# Start filtering data
sh stage2_template_all.sh
```

Step 3: Analyze the mechanism and cause localization of visual fast mapping ability
```bash
# Test scripts to quickly determine whether the code is running smoothly
sh stage3_demo.sh

# Start
sh stage3_template_all.sh
```

## License
The code is distributed under the CC-BY-NC-SA 4.0 license.

[changelog]: http://icode.baidu.com/repos/baidu_temp/acgbenchmark/vlmvisualicl/blob/master:CHANGELOG.md
