# CLIP Quadrant Experiment with `clipq`

This repository contains the `clipq` package, which provides an easy-to-use interface for experimenting with OpenAI's CLIP model by encoding image quadrants. By splitting images into quadrants and encoding each with CLIP, we can explore how the model perceives various parts of an image.

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
from clipq import CLIPQ

# Initialize the experiment
experiment = CLIPQ()

# Run the experiment on a random image from the internet
vectors = experiment.run_experiment()

# Print the obtained vectors
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

## Acknowledgments

- Thanks to OpenAI for the CLIP model.
- Inspiration drawn from various CLIP-related projects in the community.

