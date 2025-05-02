from sqlalchemy import create_engine,text
import pandas as pd 
import os 



class Database:
    
    def __init__(self):
        self.engine = create_engine("postgresql://postgres:12345@localhost:5432/gyk1")
        
    def get_customers(self):
        with self.engine.connect() as conn:
            query = text("SELECT * FROM customers")
            result = conn.execute(query)
            return result.fetchall()
        
    def get_orders(self):
        with self.engine.connect() as conn:
            query = text("SELECT * FROM orders")
            result = conn.execute(query)
            return result.fetchall()
        
    def get_products(self):
        with self.engine.connect() as conn:
            query = text("SELECT * FROM products")
            result = conn.execute(query)
            return result.fetchall()
        
        
    def get_categories(self):
         with self.engine.connect() as conn:
             query = text("SELECT * FROM categories")
             result = conn.execute(query)
             return result.fetchall()
     

    def get_order_details(self):
        with self.engine.connect() as conn:
            query = text("SELECT * FROM order_details")
            result = conn.execute(query)
            return result.fetchall()


    def get_customer_order_summary(self):
        with self.engine.connect() as conn:
            query = text("""SELECT 
                            o.customer_id, 
                            COUNT(o.order_id) AS total_order, 
                            SUM(od.unit_price * od.quantity * (1 - od.discount)) AS total_amount,
                            SUM(od.unit_price * od.quantity * (1 - od.discount)) / COUNT(DISTINCT o.order_id) AS avg_order_value, 
                            MAX(o.order_date) AS last_order_date
                        FROM order_details AS od 
                        INNER JOIN orders AS o ON o.order_id = od.order_id
                        INNER JOIN customers AS c ON o.customer_id = c.customer_id
                        GROUP BY o.customer_id;
                    """)
            result = conn.execute(query)
            return result.fetchall()
    
    def product_return(self):
        with self.engine.connect() as conn:
            query = text("""
                            SELECT 
                                c.customer_id,
                                ROUND(AVG(od.discount)::numeric, 3) AS avg_discount,
                                SUM(od.quantity) AS total_quantity,
                                SUM(od.unit_price * od.quantity * (1 - od.discount)) AS total_spending
                            FROM orders o
                            INNER JOIN customers c ON o.customer_id = c.customer_id
                            INNER JOIN order_details od ON o.order_id = od.order_id
                            GROUP BY c.customer_id
                            ORDER BY total_spending DESC;
                """)
            result = conn.execute(query)
            return result.fetchall()
    def new_product(self):
        with self.engine.connect() as conn:
            query = text("""
                        select o.customer_id, c.category_id,c.category_name, sum(od.unit_price * od.quantity * (1 - od.discount)) as total_spending
                        from products as p
                        inner join order_details as od 
                        on p.product_id = od.product_id
                        inner join categories as c
                        on c.category_id = p.category_id
                        inner join orders as o 
                        on o.order_id = od.order_id
                        group by c.category_id,o.customer_id
                        order by o.customer_id,c.category_id DESC
                """)
            result = conn.execute(query)
            return result.fetchall()
        
    def orders(self):
        with self.engine.connect() as conn:
            query = text("""
                    SELECT
                                c.customer_id,
                                o.order_date
                            FROM customers c
                            JOIN orders o ON c.customer_id = o.customer_id
                            JOIN order_details od ON o.order_id = od.order_id
                """)
            result = conn.execute(query)
            return result.fetchall()
        


    def close(self):
        self.engine.dispose()
        

"""db = Database()
result = db.get_customer_order_summary()
df = pd.DataFrame(result)
print(df)"""