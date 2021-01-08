from base64 import urlsafe_b64encode
from defusedxml.ElementTree import XMLParser  # type: ignore
from functools import partial
from hashlib import sha256
import httpx
from starlette.applications import Starlette
from starlette.config import Config
from starlette.exceptions import HTTPException
from starlette.responses import Response
from starlette.requests import Request
from starlette.routing import Route
from typing import Awaitable, Callable, Optional
from urllib.parse import parse_qsl, urlencode
from xml.etree import ElementTree


NAMESPACES = {
    "atom": "http://www.w3.org/2005/Atom",
    "content": "http://purl.org/rss/1.0/modules/content/",
    "fh": "http://purl.org/syndication/history/1.0",
    "itunes": "http://www.itunes.com/dtds/podcast-1.0.dtd",
    "slash": "http://purl.org/rss/1.0/modules/slash/",
    "sy": "http://purl.org/rss/1.0/modules/syndication/",
    "thr": "http://purl.org/syndication/thread/1.0",
    "wfw": "http://wellformedweb.org/CommentAPI/",
}

for prefix, url in NAMESPACES.items():
    ElementTree.register_namespace(prefix, url)

# If any of these tags are present in a feed, then it already seems to be an
# RFC5005 feed and clients should just use the feed directly instead of through
# this proxy.
HISTORY_TAGS = (
    "./atom:link[@rel='current']",
    "./atom:link[@rel='prev-archive']",
    "./atom:link[@rel='next-archive']",
    "./fh:complete",
    "./fh:archive",
)

config = Config(".env")

DEBUG = config("DEBUG", cast=bool, default=False)
HTTP_PROXY = config("HTTP_PROXY", default=None)

http_client = httpx.AsyncClient(
    headers={"User-Agent": "jamey@minilop.net"},
    # use forward-only mode so the proxy can see and cache even HTTPS requests
    # https://www.python-httpx.org/advanced/#proxy-mechanisms
    proxies=httpx.Proxy(url=HTTP_PROXY, mode="FORWARD_ONLY") if HTTP_PROXY else {},
)


async def feed(request: Request) -> Response:
    # Allow clients to either URL-encode query parameters in the path or append
    # them unquoted.
    url = httpx.URL(request.path_params["url"], request.query_params)

    # No matter what, make sure the feed is sorted by modification time. Even
    # the current feed must change whenever any post is modified.
    url = update_query(url, modified=None, orderby="modified")

    async with http_client.stream("GET", url) as doc:
        if doc.status_code != 200:
            raise HTTPException(doc.status_code if doc.status_code >= 400 else 502)

        if "https://api.w.org/" not in doc.links:
            raise HTTPException(403, "Not a WordPress site")

        try:
            raw_content_type = doc.headers["Content-Type"]
            content_type = raw_content_type.split(";", 1)[0].strip()
        except KeyError:
            raise HTTPException(502, "Origin didn't provide a Content-Type")

        assert doc.url is not None
        url = doc.url

        try:
            parser = XMLParser()
            async for chunk in doc.aiter_bytes():
                parser.feed(chunk)
            root = parser.close()
        except ElementTree.ParseError:
            raise HTTPException(406, "Unsupported non-XML feed format")

    feed_element = None
    if root.tag == "{" + NAMESPACES["atom"] + "}feed":
        feed_element = root
    elif root.tag == "rss":
        feed_element = root.find("./channel", NAMESPACES)

    if feed_element is None:
        raise HTTPException(406, f"Unsupported feed format {root.tag}")

    if any(feed_element.find(tag, NAMESPACES) for tag in HISTORY_TAGS):
        raise HTTPException(403, "Already an RFC5005 feed, no proxy needed")

    new_elements = []
    query = dict(parse_qsl(url.query))

    if query.get(b"order") == b"ASC":
        # This is a legit archive page. We just need to construct an
        # appropriate URL for the next-oldest archive page and insert it.
        try:
            page = int(query.get(b"paged", 1))
        except ValueError:
            raise HTTPException(400, "Invalid 'paged' parameter")

        new_elements.append(element("fh", "archive"))

        current_url = update_query(url, modified=None, order=None, paged=None)
        new_elements.append(
            element(
                "atom",
                "link",
                href=str(current_url),
                rel="current",
                type=content_type,
            )
        )

        if page > 1:
            prev_url = await hash_page(url, raw_content_type, page - 1)
            new_elements.append(
                element(
                    "atom",
                    "link",
                    href=str(prev_url),
                    rel="prev-archive",
                    type=content_type,
                )
            )

    elif b"order" in query or b"paged" in query:
        # Refuse to process non-archive feeds that don't have the newest entries.
        raise HTTPException(403)

    else:
        # We need to treat this as the main feed document, which means now we
        # need to know how many pages WordPress is going to break this feed
        # into so we can link to the last of them.
        url = update_query(url, order="ASC")
        exists = partial(page_exists, url, raw_content_type)

        page = 2
        while await exists(page):
            page *= 2

            # There's no guarantee that the server is actually honoring the
            # pagination query parameters, but we don't have a good way to
            # check that using only HEAD requests. Instead, impose an absurdly
            # high limit on how many archive pages we're willing to process.
            if page >= 65536:
                raise HTTPException(403, "Too much archives")

        # We now know that at least `page/2` pages exist but there are fewer
        # than `page`, so now we need to check all the page numbers in between.
        # Note that we're guaranteed that at least one page exists, because we
        # fetched it already. If `page` is still 2, then there is only that one
        # page, but we'll try to binary search the empty range 2..1, which will
        # immediately return 1.
        page = await binary_search(page // 2 + 1, page - 1, exists)

        if page == 1:
            # This is a complete feed, no pagination needed.
            new_elements.append(element("fh", "complete"))
        else:
            # The current document has all the posts of the final page, and
            # possibly a few extra if the number of posts per page doesn't
            # evenly divide into the total number of posts. Either way we
            # should link to the page before the final one.
            prev_url = await hash_page(url, raw_content_type, page - 1)
            new_elements.append(
                element(
                    "atom",
                    "link",
                    rel="prev-archive",
                    type=content_type,
                    href=str(prev_url),
                )
            )

    for e in reversed(new_elements):
        feed_element.insert(0, e)

    return Response(ElementTree.tostring(root), media_type=content_type)


async def hash_page(url: httpx.URL, content_type: str, page: int) -> httpx.URL:
    if page != 1:
        url = update_query(url, paged=str(page))

    async with http_client.stream("GET", url, allow_redirects=False) as doc:
        if doc.status_code != 200:
            raise HTTPException(502, f"Bad status {doc.status_code} for hashed page")

        if doc.headers.get("Content-Type") != content_type:
            raise HTTPException(
                502,
                f"Bad content type {doc.headers.get('Content-Type')} for hashed page",
            )

        contents = sha256()
        async for chunk in doc.aiter_bytes():
            contents.update(chunk)

    # This URL needs to change if this page _or any previous page_ changes.
    # Because we sort by modification timestamp, changing a post removes it
    # from the archive and appends it at the end, shifting every subsequent
    # post to a different offset. Therefore, all we really need to look at is
    # the modification timestamp and post ID of the most-recently modified post
    # on this page. But if those haven't changed, it ought to be true that
    # nothing else in this page has changed either, so hashing the entire page
    # also works, and avoids spending time and memory on XML parsing.
    return update_query(url, modified=urlsafe_b64encode(contents.digest()).decode())


async def page_exists(url: httpx.URL, content_type: str, page: int) -> bool:
    url = update_query(url, paged=str(page))
    response = await http_client.head(url, allow_redirects=False)
    if response.status_code == 404:
        return False
    if response.status_code != 200:
        raise HTTPException(502, f"Bad status {response.status_code} for exists check")
    if response.headers.get("Content-Type") != content_type:
        raise HTTPException(
            502,
            f"Bad content type {response.headers.get('Content-Type')} for exists check",
        )
    return True


def update_query(url: httpx.URL, **new_params: Optional[str]) -> httpx.URL:
    # Put the URL into a canonical form to increase cache hit rates. A generic
    # client can't make any assumptions about how the query string is used, but
    # because we only care how WordPress would interpret it, we can safely
    # de-duplicate and sort the parameters.
    params = {k: v for k, v in parse_qsl(url.query) if k.decode() not in new_params}
    params.update(
        (k.encode(), v.encode()) for k, v in new_params.items() if v is not None
    )
    query = urlencode(sorted(params.items())).encode()
    return url.copy_with(query=query)


def element(prefix: str, name: str, **kwargs: str) -> ElementTree.Element:
    return ElementTree.Element("{" + NAMESPACES[prefix] + "}" + name, attrib=kwargs)


async def binary_search(
    lo: int, hi: int, pred: Callable[[int], Awaitable[bool]]
) -> int:
    """
    Finds the largest index where pred returns True within the range lo..hi,
    inclusive. The predicate must return True for all smaller indexes and False
    for all larger ones or it's undefined which True index you'll get.

    >>> from asyncio import run
    >>> async def pred(i):
    ...     return i <= 7
    >>> run(binary_search(1, 10, pred))
    7

    If the range is empty or the predicate is True for every index in the
    range, then this function returns hi.

    >>> run(binary_search(11, 10, pred))
    10
    >>> run(binary_search(1, 5, pred))
    5

    If the predicate is not True for any index in the range, then this function
    returns lo-1.

    >>> run(binary_search(10, 20, pred))
    9
    """

    # postcondition: returns n
    # precondition: forall i, pred(i) iff i <= n
    # precondition: lo - 1 <= n <= hi
    # loop invariant: pred(lo - 1) and !pred(hi + 1)
    while lo <= hi:
        # Python has arbitrary-precision ints so we don't have to worry about
        # overflow, but let's not pay the cost for them if we don't need it.
        mid = lo + (hi - lo) // 2
        if await pred(mid):
            lo = mid + 1
        else:
            hi = mid - 1
    # Since initially lo - 1 <= hi, now hi == lo - 1. By the loop invariant
    # then, pred(hi) is True and pred(hi + 1) is False.
    return hi


app = Starlette(
    debug=DEBUG,
    routes=[
        Route("/{url:path}", feed, name="feed"),
    ],
)
