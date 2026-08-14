-- Copperleaf Kitchens Inventory Database
-- SQLite

PRAGMA foreign_keys = ON;

-- ----------------------------------------------------------
-- BRANCHES
-- Stores each restaurant branch.
-- ----------------------------------------------------------
CREATE TABLE branches (
    branch_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL,
    address         TEXT NOT NULL,
    phone           TEXT
);

-- ----------------------------------------------------------
-- STAFF
-- Stores employee information.
-- Managers have additional permissions such as writing off
-- inventory and generating waste reports.
--
-- api_token is used to identify the logged-in employee.
-- The server uses it to determine who is making requests,
-- so tool calls do not need to include staff_id or role.
-- ----------------------------------------------------------
CREATE TABLE staff (
    staff_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    branch_id       INTEGER NOT NULL,
    full_name       TEXT NOT NULL,
    email           TEXT NOT NULL UNIQUE,
    role            TEXT NOT NULL CHECK (role IN ('staff', 'manager')),
    active          INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
    api_token       TEXT NOT NULL UNIQUE,
    FOREIGN KEY (branch_id) REFERENCES branches(branch_id)
);

-- ----------------------------------------------------------
-- SUPPLIERS
-- Companies that provide inventory items.
-- ----------------------------------------------------------
CREATE TABLE suppliers (
    supplier_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL,
    contact_email   TEXT,
    phone           TEXT
);

-- ----------------------------------------------------------
-- INVENTORY_ITEMS
-- Stores the current stock available at each branch.
-- current_quantity is updated by the application whenever
-- a transaction is recorded.
-- ----------------------------------------------------------
CREATE TABLE inventory_items (
    item_id             INTEGER PRIMARY KEY AUTOINCREMENT,
    branch_id           INTEGER NOT NULL,
    supplier_id         INTEGER,
    name                TEXT NOT NULL,
    category            TEXT NOT NULL CHECK (category IN ('produce', 'protein', 'dairy', 'dry_goods', 'beverage', 'other')),
    unit                TEXT NOT NULL CHECK (unit IN ('kg', 'g', 'l', 'ml', 'unit', 'case')),
    current_quantity    REAL NOT NULL DEFAULT 0 CHECK (current_quantity >= 0),
    reorder_threshold   REAL NOT NULL DEFAULT 0,
    unit_cost           REAL NOT NULL DEFAULT 0,
    FOREIGN KEY (branch_id) REFERENCES branches(branch_id),
    FOREIGN KEY (supplier_id) REFERENCES suppliers(supplier_id)
);

-- ----------------------------------------------------------
-- INVENTORY_TRANSACTIONS
-- Keeps a history of all inventory changes.
-- Every restock, write-off, usage, or adjustment is stored
-- here instead of modifying history.
-- ----------------------------------------------------------
CREATE TABLE inventory_transactions (
    transaction_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id             INTEGER NOT NULL,
    staff_id            INTEGER NOT NULL,
    change_type         TEXT NOT NULL CHECK (change_type IN ('restock', 'write_off', 'usage', 'adjustment')),
    quantity_change     REAL NOT NULL,  -- Positive for restocks, negative for usage/write-offs.
    reason              TEXT,           -- Required by the application when writing off inventory.
    created_at          TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (item_id) REFERENCES inventory_items(item_id),
    FOREIGN KEY (staff_id) REFERENCES staff(staff_id)
);

-- ----------------------------------------------------------
-- SUPPLIER_ORDERS
-- Records orders placed with suppliers.
-- ----------------------------------------------------------
CREATE TABLE supplier_orders (
    order_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    branch_id           INTEGER NOT NULL,
    supplier_id         INTEGER NOT NULL,
    item_id             INTEGER NOT NULL,
    quantity            REAL NOT NULL CHECK (quantity > 0),
    status              TEXT NOT NULL CHECK (status IN ('pending', 'delivered', 'cancelled')) DEFAULT 'pending',
    ordered_at          TEXT NOT NULL DEFAULT (datetime('now')),
    expected_delivery   TEXT,
    FOREIGN KEY (branch_id) REFERENCES branches(branch_id),
    FOREIGN KEY (supplier_id) REFERENCES suppliers(supplier_id),
    FOREIGN KEY (item_id) REFERENCES inventory_items(item_id)
);

-- Indexes used by the most common queries.
CREATE INDEX idx_items_branch ON inventory_items(branch_id);
CREATE INDEX idx_txn_item ON inventory_transactions(item_id);
CREATE INDEX idx_txn_created ON inventory_transactions(created_at);
CREATE INDEX idx_orders_branch_status ON supplier_orders(branch_id, status);