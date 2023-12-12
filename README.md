# The Generalization Gap in Offline Reinforcement Learning

Official codebase for ["The Generalization Gap in Offline Reinforcement Learning"](https://arxiv.org/abs/2312.05742).

By [Ishita Mediratta*](https://github.com/ishitamed19), [Qingfei You*](https://github.com/YhgzXxfz), [Minqi Jiang](https://github.com/minqi), [Roberta Raileanu](https://github.com/rraileanu). [* = Equal Contribution]

```
@misc{mediratta2023generalization,
      title={The Generalization Gap in Offline Reinforcement Learning}, 
      author={Ishita Mediratta and Qingfei You and Minqi Jiang and Roberta Raileanu},
      year={2023},
      eprint={2312.05742},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

# Repository Structure

We have two sub-folders:

- **procgen**: Provides the datasets and experimental code for running experiments in the Procgen benchmark.
- **webShop**: Provides similar resources for the WebShop benchmark.

Each of these subfolders utilizes different frameworks and libraries. Therefore, please refer to the corresponding `README.md` in the respective subfolders for more information on how to setup the code, download the necessary datasets, and train or test different methods.

# License

The majority of `gen_dgrl` code is licensed under CC-BY-NC, however portions of the project are available under separate license terms: In `procgen` subfolder, code for `DT` and `online` is licensed under the MIT license. The majority of `webShop` code is licensed under MIT license (see [webshop_LICENSE.md](./webShop/webshop_LICENSE.md)), with `train_choice_[il,bcq,cql].py` files licensed under Apache 2.0 license.
