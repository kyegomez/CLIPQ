# CLIP Quadrant Experiment with `clipq`

This repository contains the `clipq` package, which provides an easy-to-use interface for experimenting with OpenAI's CLIP model by encoding image quadrants. By splitting images into quadrants and encoding each with CLIP, we can explore how the model perceives various parts of an image.

## Appreciation

- [Christopher in LAION for the idea](https://discord.com/channels/823813159592001537/824374369182416994/1158057178582753342)
- Thanks to OpenAI for the CLIP model.
- Inspiration drawn from various CLIP-related projects in the community.



## Table of Contents

- [Installation](#installation)
- [Quickstart](#quickstart)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Installation

Install the package via pip:

```bash
pip install clipq
```

## Quickstart

Here's a brief example to get you started:

```python
from clipq.main import CLIPQ

#init
test = CLIPQ(query_text="A photo of a cat")

#input, url => embed
vectors = test.run_from_url(url="https://picsum.photos/800", h_splits=3, v_splits=3)

#print
print(vectors)
```

## Contributing

1. Fork the repository on GitHub.
2. Clone the forked repository to your machine.
3. Create a new branch with an appropriate name.
4. Make your changes and commit with a meaningful commit message.
5. Push your changes to your forked repository.
6. Create a Pull Request against the original repository.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

# Todo
- Make captions using any of the following: openclip G, OpenCLIP G or siglip L or EVA G