# How to Deploy This App on a Hugging Face Space

This guide walks you through deploying your app to a Hugging Face Space, including cleaning your repository and configuring your Space.

## 1. Prepare Your Repository

If your repository contains large files (>10MB) or sensitive data, you may want to remove them before pushing to Hugging Face. For example, to remove a specific PDF:

```bash
git filter-repo --path data/pdf_docs/a-practical-guide-to-building-agents.pdf --invert-paths
```

Learn more: [Git Filter-Repo FAQ](https://www.git-tower.com/learn/git/faq/git-filter-repo)

After cleaning your repository, you will need to force push the changes to your GitHub repository:

```bash
git remote add origin git@github.com:csar95/Read-Aloud-AI.git
git push --set-upstream --force origin main
```

## 2. Deploy to Hugging Face Spaces

- Go to [Hugging Face Spaces](https://huggingface.co/spaces).
- Click **New Space** and select the desired SDK. For this app, select **Gradio**.
- Configure your Space as needed. Spaces are configured through the YAML block at the top of the README.md file at the root of the repository. See the [Spaces config reference](https://huggingface.co/docs/hub/spaces-config-reference) for details.

## 3. Automate Deployments (Optional)

You can use GitHub Actions to automate deployments to Spaces. See the [Spaces GitHub Actions documentation](https://huggingface.co/docs/hub/spaces-github-actions) for setup instructions.
