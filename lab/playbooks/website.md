---
id: website
title: Website
label: Website
keywords: [website, site, web, landing page, homepage, portfolio, blog, pages, web app, webapp, dashboard, booking, restaurant, company site]
---

Internal guidance for the team. Adapt it to the actual site; it is not a feature list.

## Considerations

- Target audience, and what they come to the site to do.
- User journeys: the two or three paths that matter most.
- Pages and navigation: how a visitor moves between them.
- Responsive behaviour across phone, tablet and desktop.
- Accessibility: semantic structure, contrast, keyboard use, image descriptions.
- Content: who supplies it, and what placeholder content is acceptable meanwhile.
- Backend requirements: decide whether the site needs one at all.
- Data persistence, only when something must be remembered.
- SEO, where being found through search matters to the business.
- Authentication and security, where accounts or private data exist.

## Typical Development Phases

1. Site foundation and shared layout.
2. Main landing or home page.
3. Core content pages.
4. Interactive features or forms.
5. Responsive and accessibility pass.
6. Final review.

## Quality Requirements

- Every page is reachable through the navigation.
- Layout remains usable from narrow phones to wide desktops.
- Pages use semantic HTML and descriptive titles.
- User input is validated on the server when a server exists.
- No secrets or credentials appear in page source or repository files.

## Common Edge Cases

- Broken internal links and missing pages.
- Very long text, or an empty list, in a layout built for typical content.
- Missing images.
- Form submission with empty, oversized or malicious input.
- JavaScript unavailable or failing, where content should still display.
- Refreshing or linking directly to an inner page.

## Suggested Acceptance Checks

- Each page responds successfully and contains its key content.
- All internal navigation links resolve.
- Forms reject invalid input and show a clear message.
- A missing page produces a helpful not-found response.
- Each page has a unique title and a single main heading.
