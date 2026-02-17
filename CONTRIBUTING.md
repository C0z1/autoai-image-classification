# Contributing to AutoAI Image Classification

Thank you for your interest in contributing! 🎉

## How to Contribute

### Reporting Issues
1. Check if the issue already exists
2. Use the issue template
3. Provide detailed information:
   - Python version
   - TensorFlow version
   - Error messages
   - Steps to reproduce

### Suggesting Enhancements
1. Open an issue with tag `enhancement`
2. Describe the feature
3. Explain use cases
4. Provide examples if possible

### Pull Requests

#### Setup Development Environment
```bash
git clone https://github.com/yourusername/autoai-image-classification.git
cd autoai-image-classification
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

#### Making Changes
1. Fork the repository
2. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Make your changes
4. Test thoroughly
5. Commit with clear messages:
   ```bash
   git commit -m "Add: descriptive commit message"
   ```
6. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
7. Open a Pull Request

#### Code Style
- Follow PEP 8 for Python code
- Use meaningful variable names
- Add comments for complex logic
- Include docstrings for functions
- Keep functions focused and small

#### Testing
- Test with different CNN models
- Verify CSV output format
- Check edge cases
- Test with various image sizes

### Areas for Contribution

#### High Priority
- [ ] Add support for more CNN models
- [ ] Implement data augmentation
- [ ] Add unit tests
- [ ] Create Docker container
- [ ] Add CI/CD pipeline

#### Medium Priority
- [ ] Web interface for feature extraction
- [ ] Automated screenshot capture
- [ ] Performance benchmarks
- [ ] Multi-GPU support
- [ ] Progress bars for processing

#### Documentation
- [ ] Video tutorials
- [ ] More examples
- [ ] Troubleshooting guide
- [ ] FAQ section
- [ ] API documentation

### Commit Message Guidelines

Use conventional commits:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes
- `refactor:` Code refactoring
- `test:` Adding tests
- `chore:` Maintenance tasks

Examples:
```
feat: add EfficientNetV2 support
fix: resolve memory leak in batch processing
docs: update Watson Studio setup guide
```

### Review Process

1. Maintainer reviews PR
2. Feedback provided if needed
3. Changes requested or approved
4. Merged into main branch
5. Released in next version

### Questions?

- Open an issue with tag `question`
- Check existing discussions
- Review documentation first

Thank you for contributing! 🚀
