import sys, os
sys.path.insert(0, r"d:\Users\gabep\Desktop\sweetweb")
os.chdir(r"d:\Users\gabep\Desktop\sweetweb")

from web_app import app

if __name__ == "__main__":
    print("Open: http://localhost:5002")
    app.run(debug=False, use_reloader=False, host="0.0.0.0", port=5002)
