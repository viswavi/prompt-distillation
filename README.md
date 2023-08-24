# prompt2model

[![PyPI version](https://badge.fury.io/py/prompt2model.svg)](https://badge.fury.io/py/prompt2model)
![Github Actions CI tests](https://github.com/neulab/prompt2model/actions/workflows/ci.yml/badge.svg)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)

`Prompt2Model` is a system that takes a natural
language task description (like the prompts used for
LLMs such as ChatGPT) to train a small
special-purpose model that is conducive for deployment.

<img width="360" alt="prompt2model_teaser" src="https://github.com/neulab/prompt2model/assets/2577384/39ca466a-5355-4d82-8312-303e52ba2bca">

## Quick Start
```bash
pip install .
```

Our current `prompt2model` implementation uses
the OpenAI API. Accordingly, you need to:

- Sign up on the OpenAI website and obtain an
OpenAI API key.

- Set
the environment variable
`OPENAI_API_KEY` to your API key by running
the following command in your terminal:

```bash
export OPENAI_API_KEY=<your key>
```

You can then run
```
python cli_demo.py
```
to 
create a small model from a prompt, as shown in 
the demo video below. This script must be run on a
device with an internet connection to access the OpenAI
API. For best results, run
this script on a device with a GPU for training
your model.

## Demo
https://github.com/neulab/prompt2model/assets/2577384/8d73394b-3028-4a0b-bdc3-c127082868f2




## How to Write a Good Prompt

A good prompt can make the generated dataset
follow exactly the format of demonstrations.
It contains the instruction and few-shot examples.

The instruction should contain the following:

1. The exact format description for the input
and output, i.e., a string, a dictionary, or whatever.
2. The exact contents of each part of the
input and their relationship as possible as you can.
3. The range of possible input. For example,
"And the question can range from Math, Cultural,
Social, Geometry, Biology, History, Sports, Technology,
Science, and so on."

The few-shot examples should contain the following:

1. Use `=` rather than other ambiguous symbols like `:`.
2. Avoid unnecessary line breaks at the beginning.
For example, `input=""` is better than breaking
the line after `=`.
3. Use `input` rather than `Input`, `ouput` is
preferable likewise.
4. Wrap the `input` and `output` into a string with `“”`.

Though the examples are optional, we strongly
suggest including them to guide the format and
content for the generator.

Also, we recommend providing several precise examples
in the specified format and inquiring with ChatGPT
about the format and scope of your examples.


## Components

The `prompt2model` package is composed
of several components, each designed
to fulfill a specific purpose. To gain
a comprehensive understanding of how to
utilize each component effectively,
please consult the `readme.md` file
situated in the directory of the respective
component. These files can be found at
`./prompt2model/<component>/readme.md`.
They provide detailed information and
instructions on customizing and maximizing
the functionality of each
component within the package.

## Contribution

If you're interested in contributing
to the `prompt2model` project, please
refer to [CONTRIBUTING.md](CONTRIBUTING.md),
or reach out to [@vijaytarian](https://twitter.com/vijaytarian)
and [@ChenytangZhao](https://twitter.com/ChenytangZhao).


## Cite

We have [written a paper describing Prompt2Model in detail](https://arxiv.org/abs/YYMM.XXXXX).

If you use Prompt2Model in your research, please cite our paper:
```
@misc{prompt2model,
      title={Prompt2Model: Generating Deployable Models from Natural Language Instructions}, 
      author={Vijay Viswanathan and Chenyang Zhao and Amanda Bertsch and Tongshuang Wu and Graham Neubig},
      year={2023},
      eprint={YYMM.XXXXX},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
