"""MkDocs hooks: publish the repo-root docs pages and send links to code at GitHub."""
import posixpath
import re
from pathlib import Path

from mkdocs.structure.files import File

REPO = "https://github.com/atiladeokegab/biopsy-scan/blob/main/"
# Repo-root page -> its path in the site.
ROOT = {"README.md": "index.md", "RESULTS.md": "RESULTS.md", "ARCHITECTURE.md": "ARCHITECTURE.md",
        "CONTRIBUTING.md": "CONTRIBUTING.md", "CODE_OF_CONDUCT.md": "CODE_OF_CONDUCT.md",
        "SECURITY.md": "SECURITY.md"}
SITE_TO_REPO = {v: k for k, v in ROOT.items()}
LINK = re.compile(r"(\]\()([^)\s]+)(\))")


def on_files(files, config):
    root = Path(config["config_file_path"]).parent
    for repo_path, site_path in ROOT.items():
        files.append(File.generated(config, site_path, abs_src_path=str(root / repo_path)))
    return files


def rewrite(target, page_repo, page_site):
    """Map a link written for GitHub to the right place in the built site."""
    if re.match(r"^[a-z]+:|^#|^/", target):
        return target
    path, _, anchor = target.partition("#")
    repo_path = posixpath.normpath(posixpath.join(posixpath.dirname(page_repo), path))
    if repo_path in ROOT:
        site = ROOT[repo_path]
    elif repo_path.startswith("docs/") and (repo_path.endswith(".md") or repo_path.startswith("docs/assets/")):
        site = repo_path[len("docs/"):]
    else:
        return REPO + repo_path + ("#" + anchor if anchor else "")
    rel = posixpath.relpath(site, posixpath.dirname(page_site) or ".")
    return rel + ("#" + anchor if anchor else "")


def on_page_markdown(markdown, page, config, files):
    site = page.file.src_uri
    repo = SITE_TO_REPO.get(site, "docs/" + site)
    return LINK.sub(lambda m: m[1] + rewrite(m[2], repo, site) + m[3], markdown)


def _self_check():
    assert rewrite("docs/GPU.md#x", "README.md", "index.md") == "GPU.md#x"
    assert rewrite("RESULTS.md", "docs/JUDGING.md", "JUDGING.md") == "RESULTS.md"
    assert rewrite("../RESULTS.md", "docs/JUDGING.md", "JUDGING.md") == "RESULTS.md"
    assert rewrite("core/", "README.md", "index.md") == REPO + "core"
    assert rewrite("docs/assets/race.gif", "README.md", "index.md") == "assets/race.gif"
    assert rewrite("https://x.org", "README.md", "index.md") == "https://x.org"


if __name__ == "__main__":
    _self_check()
    print("ok")
