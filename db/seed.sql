-- Copperleaf Kitchens - Sample Data
-- Populates the database with example records for testing.

-- ----------------------------------------------------------
-- BRANCHES
-- ----------------------------------------------------------
INSERT INTO branches (branch_id, name, address, phone) VALUES
    (1, 'Copperleaf Downtown', '120 Market St, Alexandria', '+20-3-555-0101'),
    (2, 'Copperleaf Harbor',   '45 Corniche Rd, Alexandria', '+20-3-555-0102');

-- ----------------------------------------------------------
-- STAFF
-- Includes managers and staff members for permission testing.
-- api_token is used by the server to identify the logged-in user.
-- ----------------------------------------------------------
INSERT INTO staff (staff_id, branch_id, full_name, email, role, active, api_token) VALUES
    (1, 1, 'Mona Farid',    'mona.farid@copperleaf.com',    'manager', 1, 'tok_mona_mgr_9f2a'),
    (2, 1, 'Youssef Adel',  'youssef.adel@copperleaf.com',  'staff',   1, 'tok_youssef_stf_c71b'),
    (3, 2, 'Salma Nabil',   'salma.nabil@copperleaf.com',   'manager', 1, 'tok_salma_mgr_4d8e'),
    (4, 2, 'Karim Fathy',   'karim.fathy@copperleaf.com',   'staff',   1, 'tok_karim_stf_1a6f'),
    (5, 1, 'Hana Zaki',     'hana.zaki@copperleaf.com',     'staff',   0, 'tok_hana_stf_e03c'); -- Inactive account.

-- ----------------------------------------------------------
-- SUPPLIERS
-- ----------------------------------------------------------
INSERT INTO suppliers (supplier_id, name, contact_email, phone) VALUES
    (1, 'Nile Fresh Produce',      'orders@nilefresh.com',    '+20-2-555-0201'),
    (2, 'Delta Dairy Co.',         'sales@deltadairy.com',    '+20-2-555-0202'),
    (3, 'Coastal Seafood & Meats', 'orders@coastalmeats.com', '+20-2-555-0203');

-- ----------------------------------------------------------
-- INVENTORY ITEMS
-- Includes both normal and low-stock inventory items.
-- ----------------------------------------------------------
INSERT INTO inventory_items (
    item_id,
    branch_id,
    supplier_id,
    name,
    category,
    unit,
    current_quantity,
    reorder_threshold,
    unit_cost
) VALUES
    (1, 1, 1, 'Roma Tomatoes',   'produce',   'kg',   4.5, 10.0, 1.20), -- LOW STOCK
    (2, 1, 1, 'Yellow Onions',   'produce',   'kg',  22.0, 8.0,  0.60),
    (3, 1, 2, 'Whole Milk',      'dairy',     'l',   15.0, 12.0, 0.95),
    (4, 1, 3, 'Chicken Breast',  'protein',   'kg',   9.0, 15.0, 3.40), -- LOW STOCK
    (5, 1, 1, 'Basmati Rice',    'dry_goods', 'kg',  40.0, 10.0, 0.85),
    (6, 2, 3, 'Salmon Fillet',   'protein',   'kg',   6.0, 5.0,  8.75),
    (7, 2, 2, 'Feta Cheese',     'dairy',     'kg',   3.0, 4.0,  4.10), -- LOW STOCK
    (8, 2, 1, 'Cucumbers',       'produce',   'kg',  18.0, 6.0,  0.75),
    (9, 2, 1, 'Sparkling Water', 'beverage',  'case', 25.0, 10.0, 6.00),
    (10, 1, 3, 'Wagyu Beef Ribeye', 'protein', 'kg',  25.0, 5.0, 85.00); -- HIGH VALUE ITEM (unit cost > $50)


-- ----------------------------------------------------------
-- INVENTORY TRANSACTIONS
-- Sample transaction history used for reporting and testing.
-- ----------------------------------------------------------
INSERT INTO inventory_transactions (
    item_id,
    staff_id,
    change_type,
    quantity_change,
    reason,
    created_at
) VALUES
    (1, 2, 'usage',     -3.5, NULL,                  '2026-07-20 09:00:00'),
    (1, 1, 'write_off', -2.0, 'spoiled_before_use',  '2026-07-22 14:15:00'),
    (2, 2, 'usage',     -6.0, NULL,                  '2026-07-21 10:30:00'),
    (3, 1, 'write_off', -3.0, 'past_expiry',         '2026-07-23 08:45:00'),
    (4, 2, 'usage',     -5.0, NULL,                  '2026-07-24 11:00:00'),
    (5, 2, 'restock',   20.0, NULL,                  '2026-07-19 07:00:00'),
    (6, 3, 'write_off', -1.5, 'damaged_in_delivery', '2026-07-25 13:20:00'),
    (7, 3, 'write_off', -1.0, 'spoiled_before_use',  '2026-07-26 16:00:00'),
    (8, 4, 'usage',     -4.0, NULL,                  '2026-07-27 09:15:00'),
    (9, 4, 'restock',   15.0, NULL,                  '2026-07-18 07:30:00');

-- ----------------------------------------------------------
-- SUPPLIER ORDERS
-- Orders with different statuses for testing.
-- ----------------------------------------------------------
INSERT INTO supplier_orders (
    branch_id,
    supplier_id,
    item_id,
    quantity,
    status,
    ordered_at,
    expected_delivery
) VALUES
    (1, 1, 1, 30.0, 'pending',   '2026-07-29 08:00:00', '2026-08-02'),
    (1, 3, 4, 25.0, 'pending',   '2026-07-30 08:00:00', '2026-08-01'),
    (1, 2, 3, 10.0, 'delivered', '2026-07-20 08:00:00', '2026-07-21'),
    (2, 2, 7, 12.0, 'pending',   '2026-07-29 09:00:00', '2026-08-01'),
    (2, 1, 8, 20.0, 'cancelled', '2026-07-15 08:00:00', '2026-07-17'); -- Cancelled order.