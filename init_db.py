import sqlite3

connection = sqlite3.connect('gardens.db')
    
cur = connection.cursor()

cur.execute("INSERT INTO user (id, name, email, password) VALUES (?, ?, ?, ?)",
            (1, 'Giovanna', 'giovanna@gmail.com', '12345')
            )

connection.commit()
connection.close()