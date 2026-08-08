"""
Purchase Order Generation (Methodology step 4).

When a product's current stock falls to/below its reorder point, this
module drafts a PO sized at the Economic Order Quantity, selects the best
supplier by a weighted score of price, lead time, and reliability, and
attaches a chain-of-thought explanation. POs are created with status
"pending_approval" -- they are routed for human approval, never auto-sent.
"""
from dataclasses import dataclass
from datetime import date, timedelta


@dataclass
class DraftPO:
    product_sku: str
    supplier_name: str
    quantity: int
    unit_cost: float
    total_cost: float
    expected_delivery: date
    reason: str


def score_supplier(supplier, base_unit_cost: float) -> float:
    """Weighted score: lower price & lead time is better, higher reliability is better.
    Normalized so a higher score == a better supplier choice."""
    price_component = 1 / max(supplier.price_index, 0.01)          # cheaper => higher
    speed_component = 1 / max(supplier.avg_lead_time_days, 1)       # faster => higher
    reliability_component = supplier.reliability_score              # 0-1, higher is better
    return (0.4 * price_component) + (0.25 * speed_component) + (0.35 * reliability_component)


def choose_best_supplier(candidate_suppliers: list, base_unit_cost: float):
    if not candidate_suppliers:
        return None
    return max(candidate_suppliers, key=lambda s: score_supplier(s, base_unit_cost))


def generate_draft_po(product, plan, supplier, today: "date | None" = None) -> DraftPO:
    """product: models.Product; plan: inventory_optimizer.InventoryPlan; supplier: models.Supplier"""
    today = today or date.today()
    qty = max(plan.economic_order_qty, plan.reorder_point - plan.current_stock, 1)
    unit_cost = product.unit_cost * supplier.price_index
    total_cost = round(qty * unit_cost, 2)
    expected_delivery = today + timedelta(days=supplier.avg_lead_time_days)

    reason = (
        f"Trigger: current stock ({plan.current_stock}) is at or below the reorder point "
        f"({plan.reorder_point}). {plan.reasoning} "
        f"Selected {supplier.name} (reliability {supplier.reliability_score:.0%}, "
        f"avg lead time {supplier.avg_lead_time_days}d, price index {supplier.price_index:.2f}) "
        f"as the best-scoring available supplier. Ordering the Economic Order Quantity "
        f"({plan.economic_order_qty} units) balances ordering costs against holding costs."
    )

    return DraftPO(
        product_sku=product.sku,
        supplier_name=supplier.name,
        quantity=qty,
        unit_cost=round(unit_cost, 2),
        total_cost=total_cost,
        expected_delivery=expected_delivery,
        reason=reason,
    )
