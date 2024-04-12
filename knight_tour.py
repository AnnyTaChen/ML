N = 8

# 初始化棋盤 -1代表沒被訪問過 被訪問過即改變 N:代表棋盤行列
def init_board(N):
    board = [[-1 for i in range(N)] for j in range(N)]
    return board

# 檢查x,y座標是否在棋盤上且未被訪問
def is_valid(x, y, board):
    return x >= 0 and y >= 0 and x < N and y < N and board[x][y] == -1

# 使用回溯法解決騎士巡遊問題
def solve_kt():
    board = init_board(N)
    
    # 騎士的移動方向index有對應
    move_x = [2, 1, -1, -2, -2, -1, 1, 2]
    move_y = [1, 2, 2, 1, -1, -2, -2, -1]
# 遞歸地嘗試不同的移動路徑
def solve_kt_util(x, y, movei, board, move_x, move_y):
    if movei == N * N:
        return True
    
    # 嘗試所有下一步
    for i in range(8):
        next_x = x + move_x[i]
        next_y = y + move_y[i]
        if is_valid(next_x, next_y, board):
            board[next_x][next_y] = movei
            if solve_kt_util(next_x, next_y, movei + 1, board, move_x, move_y):
                return True
            # 如果移動不成功，回溯
            board[next_x][next_y] = -1
    return False

solve_kt()
