---
id: ecommerce
title: E-commerce
label: Website / E-commerce
keywords: [shop, store, ecommerce, e-commerce, marketplace, cart, checkout, catalog, catalogue, inventory, products for sale, sell, selling, orders, retail, online store]
---

Internal guidance for the team. Adapt it to the actual business; it is not a feature
list. A small shop may need only a catalog and contact details.

## Considerations

- Catalog: what is sold, and which details customers need before buying.
- Categories, only if the range is large enough to need them.
- Search and filtering, if customers will struggle to browse.
- Product details: images, price, availability, variants.
- Cart: whether customers buy online at all, or enquire instead.
- Orders: what happens after purchase, and who handles it.
- Inventory: whether stock is tracked, and what happens when it runs out.
- Payment integration, only when the user asks for it.
- Customer data security: collect the minimum, and protect what is collected.
- Currency, tax, delivery areas and legal pages, where they apply.

## Typical Development Phases

1. Store foundation and shared layout.
2. Home page.
3. Product catalog.
4. Product details.
5. Cart and order flow, when requested.
6. Business and contact information.
7. Final review.

## Quality Requirements

- Prices are stored and calculated without floating-point rounding errors.
- Stock status shown to a customer reflects stored data.
- Totals are calculated on the server, never trusted from the client.
- Queries are parameterized, so search input cannot alter them.
- Customer data is not exposed in URLs, logs or page source.

## Common Edge Cases

- A search with no matches, which needs a clear empty state.
- A product that is out of stock, or that goes out of stock during purchase.
- Zero, negative or absurdly large quantities.
- A cart containing a product that has since been removed or repriced.
- Special characters in search terms.
- Very long product names and missing product images.

## Suggested Acceptance Checks

- The catalog lists stored products with name, price and availability.
- Search is case-insensitive, and shows a clear message when nothing matches.
- A product detail page shows the correct product, and a missing one is handled.
- Cart totals, when a cart exists, are correct after adding, changing and removing items.
- Invalid quantities are rejected.
