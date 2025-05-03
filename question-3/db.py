from sqlalchemy import create_engine,text
import pandas as pd 
import os 



class Database:
    
    def __init__(self):
        self.engine = create_engine("postgresql://postgres:2468aybuke@localhost:5432/Gyk1northwinds")
        
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
                        SELECT 
                            o.customer_id, 
                            c.category_id, 
                            c.category_name, 
                            p.product_id,
                            p.product_name,
                            SUM(od.unit_price * od.quantity * (1 - od.discount)) AS total_spending
                        FROM products AS p
                        INNER JOIN order_details AS od 
                            ON p.product_id = od.product_id
                        INNER JOIN categories AS c
                            ON c.category_id = p.category_id
                        INNER JOIN orders AS o 
                            ON o.order_id = od.order_id
                        GROUP BY o.customer_id, c.category_id, c.category_name, p.product_id, p.product_name
                        ORDER BY o.customer_id, c.category_id DESC;

                """)
            result = conn.execute(query)
            return result.fetchall()
        
    def orders(self):
        with self.engine.connect() as conn:
            query = text("""
                SELECT DISTINCT
                c.customer_id,
                DATE(o.order_date) AS order_day
                FROM customers c
                JOIN orders o ON c.customer_id = o.customer_id
                ORDER BY c.customer_id, order_day;
                """)
            result = conn.execute(query)
            return result.fetchall()
    
    def user_product_interactions(self):
        with self.engine.connect() as conn:
            query = text("""
                        SELECT
                            o.customer_id AS user_id,
                            od.product_id,
                            COUNT(*) AS rating
                        FROM orders o
                        JOIN order_details od ON o.order_id = od.order_id
                        GROUP BY o.customer_id, od.product_id
                    """)
            result = conn.execute(query)
            return result.fetchall()
        


    def close(self):
        self.engine.dispose()
        

"""db = Database()
result = db.get_customer_order_summary()
df = pd.DataFrame(result)
print(df)"""