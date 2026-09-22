# Splash page

The public landing page lives in [`site/`](../site/). It uses plain HTML, CSS,
and JavaScript and needs no build step or package installation. Keep all public
assets inside that directory and use relative asset paths so the page works at
the repository's `/seamstress/` URL.

## Preview

From the repository root:

```sh
python3 -m http.server 4173 --bind 127.0.0.1 --directory site
```

Visit <http://127.0.0.1:4173/>. Stop the server with Ctrl-C when finished.

## GitHub Pages

In the repository's **Settings → Pages → Build and deployment**, choose
**GitHub Actions** as the source. Limit the `github-pages` environment's deployment
branches to `main`.

Once the page and [workflow](../.github/workflows/pages.yml) are on `main`, pushes
that change `site/` or the workflow deploy the page automatically. For the first
deployment or a retry, run **GitHub Pages** from the Actions tab with `main`
selected. Other branches cannot deploy through this workflow.

The expected public address is <https://heresalexandria.github.io/seamstress/>.
The deployment job reports the actual address after publication. Only `site/`
is uploaded; app code, research, and original media are outside the artifact.
The workflow rejects symbolic links in the public directory and uses pinned
official Pages actions with deployment permissions limited to the deploy job.

Download buttons use the stable GitHub Releases asset aliases for Apple silicon
and Intel Macs, so a new app release does not require a page edit. Site-only
pull requests should carry `no-release`.

## Assets and content

The page reuses the app icon and self-hosts its Instrument Serif, Manrope, and
IBM Plex Mono fonts. Their licenses are included in `site/assets/licenses/`.
The original SVG landscape is an explanatory illustration, not an app screenshot
or a processing result. The before/after buttons only change this illustration.
Downloads, navigation, and the FAQ remain usable without JavaScript.

When updating copy, describe the latest published app, not unreleased branches.
Check both installer links, keyboard navigation, reduced motion, the no-JavaScript
fallback, and phone/tablet/desktop layouts before merging. Preview images and
browser check output belong in ignored `output/`, outside the public directory.
