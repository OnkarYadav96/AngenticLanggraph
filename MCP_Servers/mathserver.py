from mcp.server.fastmcp import FastMCP

mcp=FastMCP("Math")

@mcp.tool()
def add(a:int,b:int)->int:
    """_summary_
    add two numbers"""
    return a+b;

@mcp.tool()
def multiple(a:int,b:int)->int:
    """Multiply two numbers"""
    return a*b

# The transport="stdio" argument tells the server to:
#Used standred input/output (stdin and stdout) to receive and respond to tool function call

if __name__=="__main__":
    mcp_run(transport="stdio")